# -*- coding: utf-8 -*-
"""
mini_engine.py —— 从零实现的 KV Cache 推理引擎（库模块）

与 nanochat 自带 `nanochat.engine.Engine` **接口兼容**，可直接替换使用：

    # 原版
    from nanochat.engine import Engine
    engine = Engine(model, tokenizer)

    # 本实现（同样的调用方式）
    from nanochat.mini_engine import MiniEngine
    engine = MiniEngine(model, tokenizer)

    for token_column, token_masks in engine.generate(tokens, num_samples=1,
                                                      max_tokens=256,
                                                      temperature=0.8, top_k=None):
        ...

兼容接口：
    - generate(tokens, num_samples=1, max_tokens=256, temperature=1.0, top_k=None, seed=42)
        流式生成器，每步 yield (token_column, token_masks)：长度均为 num_samples
    - generate_batch(tokens, num_samples=1, **kwargs)
        非流式，返回 (results, masks)；终止 token（assistant_end / bos）不计入结果

额外提供（用于对照与实验）：
    - generate_cached(...)  单样本、带 KV Cache 的增量解码（prefill + decode，逻辑最清晰）
    - generate_naive(...)   朴素生成：每步喂全部历史、重算 K/V（对照实现）
    - compare(...)          cache vs naive：正确性（贪心逐 token 一致）+ prefill/decode 耗时
    - chat(prompt, ...)     对话（渲染 prompt → 生成 → 解码）

实现要点：
    1. prefill：prompt 的 K/V 一次算完写入 KV Cache
    2. decode ：每步只喂 1 个新 token，历史 K/V 从 cache 复用（不重算）
    3. GQA：cache 的 KV 头数取 config.n_kv_head（可与 Q 头数不同）
    4. 位置编码：由模型内部按 cache 位置（cache.get_pos()）取 RoPE，天然是绝对位置
"""

import time

import torch

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache


class MiniEngine:
    """从零实现的 KV Cache 推理引擎（接口兼容 nanochat.engine.Engine）"""

    def __init__(self, model, tokenizer, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

        # KV Cache 维度从模型配置取（GQA：KV 头数可能小于 Q 头数）
        m = self.model.config
        self.kv_kwargs = {
            "num_heads": getattr(m, "n_kv_head", m.n_head),
            "head_dim": m.n_embd // m.n_head,
            "num_layers": m.n_layer,
        }
        self.max_seq_len = m.sequence_len                       # 模型 RoPE 缓存上限
        self.stop_ids = {
            self.tokenizer.encode_special("<|assistant_end|>"),
            self.tokenizer.get_bos_token_id(),
        }

    # ==================================================================
    # 兼容接口 ①：流式生成（对应 Engine.generate）
    # ==================================================================
    def generate(self, tokens, num_samples=1, max_tokens=256, temperature=1.0, top_k=None, seed=42):
        """
        流式生成：每步 yield (token_column, token_masks)
            token_column: list[int]，长度 = num_samples（每个样本当前步的 token）
            token_masks : list[int]，长度 = num_samples（1 = 生成部分）
        与 nanochat Engine.generate 接口兼容，chat_cli 等脚本可直接切换使用。
        """
        self.model.eval()
        tokens = list(tokens)
        if len(tokens) >= self.max_seq_len:
            tokens = tokens[-(self.max_seq_len - 1):]
        B = max(1, int(num_samples))
        cache_len = max(min(len(tokens) + int(max_tokens), self.max_seq_len), len(tokens))

        cache = KVCache(
            batch_size=B, seq_len=cache_len, device=self.device,
            dtype=COMPUTE_DTYPE, **self.kv_kwargs,
        )
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)

        with torch.no_grad():
            # prefill：同一 prompt 复制 B 份，一次算完写入 cache
            x = torch.tensor([tokens] * B, dtype=torch.long, device=self.device)
            logits = self.model.forward(x, kv_cache=cache)
            finished = [False] * B
            n_gen = 0
            for _ in range(int(max_tokens)):
                if len(tokens) + n_gen + 1 > cache_len:
                    break
                nxt = self._sample_batch(logits[:, -1, :], temperature, top_k, rng)
                yield nxt, [1] * B
                n_gen += 1
                for i, t in enumerate(nxt):
                    if t in self.stop_ids:
                        finished[i] = True
                if all(finished):
                    break
                # decode：只喂 B 个新 token，历史 K/V 复用
                x = torch.tensor([[t] for t in nxt], dtype=torch.long, device=self.device)
                logits = self.model.forward(x, kv_cache=cache)

    # ==================================================================
    # 兼容接口 ②：非流式批量生成（对应 Engine.generate_batch）
    # ==================================================================
    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """返回 (results, masks)；终止 token（assistant_end / bos）不计入结果。"""
        results = [list(tokens) for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token in self.stop_ids:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        return results, masks

    # ==================================================================
    # 单样本、带 KV Cache 的增量解码（逻辑最清晰，用于阅读与对比实验）
    # ==================================================================
    def generate_cached(self, tokens, max_new=50, temperature=0.0, top_k=None,
                        seed=42, stop_at_end=True):
        """prefill + decode：prefill 一次算完 prompt，decode 每步只喂 1 个新 token"""
        self.model.eval()
        tokens = list(tokens)
        if len(tokens) >= self.max_seq_len:
            tokens = tokens[-(self.max_seq_len - 1):]
        cache_len = max(min(len(tokens) + max_new, self.max_seq_len), len(tokens))

        cache = KVCache(
            batch_size=1, seq_len=cache_len, device=self.device,
            dtype=COMPUTE_DTYPE, **self.kv_kwargs,
        )
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)

        out = tokens
        with torch.no_grad():
            # --- prefill ---
            x = torch.tensor([tokens], dtype=torch.long, device=self.device)
            logits = self.model.forward(x, kv_cache=cache)
            # --- decode ---
            for _ in range(max_new):
                if len(out) >= cache_len:
                    break
                nxt = self._sample_batch(logits[:, -1, :], temperature, top_k, rng)[0]
                out.append(nxt)
                if stop_at_end and nxt in self.stop_ids:
                    break
                x = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
                logits = self.model.forward(x, kv_cache=cache)   # 只算 1 个 token
        return out

    # ==================================================================
    # 对照实现：朴素生成（无 cache，每步喂全部历史 → 重算 K/V）
    # ==================================================================
    def generate_naive(self, tokens, max_new=50, temperature=0.0, top_k=None,
                       seed=42, stop_at_end=True):
        self.model.eval()
        tokens = list(tokens)
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        out = tokens
        with torch.no_grad():
            for _ in range(max_new):
                if len(out) >= self.max_seq_len:
                    break
                x = torch.tensor([out], dtype=torch.long, device=self.device)
                logits = self.model.forward(x)                    # 全量重算
                nxt = self._sample_batch(logits[:, -1, :], temperature, top_k, rng)[0]
                out.append(nxt)
                if stop_at_end and nxt in self.stop_ids:
                    break
        return out

    def _sample_batch(self, logits, temperature, top_k, rng):
        """对 [B, V] 的 logits 逐行采样；temperature<=0 → 贪心。返回 list[int]（长度 B）"""
        if temperature is None or temperature <= 0:
            return [int(t) for t in torch.argmax(logits, dim=-1)]
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            logits = logits.clone()
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        return [int(t) for t in torch.multinomial(probs, num_samples=1, generator=rng).squeeze(-1)]

    # ==================================================================
    # 对比实验：cache vs naive
    # ==================================================================
    def compare(self, tokens, max_new=50):
        """
        公平对比：贪心解码 + 强制生成满额（忽略停止 token）
        分开统计 prefill / decode，突出 KV Cache 的真正价值（decode 每步只算 1 个 token）
        """
        tokens = list(tokens)
        if len(tokens) >= self.max_seq_len:
            tokens = tokens[-(self.max_seq_len - 1):]
        cache_len = max(min(len(tokens) + max_new, self.max_seq_len), len(tokens))

        # ① cache 版：prefill + decode
        cache = KVCache(batch_size=1, seq_len=cache_len, device=self.device,
                        dtype=COMPUTE_DTYPE, **self.kv_kwargs)
        with torch.no_grad():
            x = torch.tensor([tokens], dtype=torch.long, device=self.device)
            t0 = time.time()
            logits = self.model.forward(x, kv_cache=cache)          # prefill
            t_prefill = time.time() - t0

            out_cache = list(tokens)
            t0 = time.time()
            n_steps = 0
            for _ in range(max_new):
                if len(out_cache) >= cache_len:
                    break
                nxt = int(torch.argmax(logits[0, -1]).item())
                out_cache.append(nxt)
                x = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
                logits = self.model.forward(x, kv_cache=cache)      # decode：1 个 token
                n_steps += 1
            t_decode = time.time() - t0

        # ② naive 版：每步全量重算
        with torch.no_grad():
            out_naive = list(tokens)
            t0 = time.time()
            n_naive = 0
            for _ in range(max_new):
                if len(out_naive) >= self.max_seq_len:
                    break
                x = torch.tensor([out_naive], dtype=torch.long, device=self.device)
                logits = self.model.forward(x)                      # 全量
                out_naive.append(int(torch.argmax(logits[0, -1]).item()))
                n_naive += 1
            t_naive = time.time() - t0

        same = out_cache == out_naive
        ms_cache = 1000 * t_decode / max(1, n_steps)
        ms_naive = 1000 * t_naive / max(1, n_naive)
        total_cache = t_prefill + t_decode

        print("=" * 58)
        print("KV Cache 对比实验（贪心解码，强制生成满额）")
        print("=" * 58)
        print(f"prompt tokens      : {len(tokens)}")
        print(f"新生成 tokens      : {n_steps}")
        print(f"结果一致           : {same}   ← 正确性：cache 不改变结果")
        print("-" * 58)
        print(f"[cache 版] prefill : {t_prefill * 1000:.1f} ms（prompt 一次算完）")
        print(f"[cache 版] decode  : {t_decode * 1000:.1f} ms / {n_steps} 步 = {ms_cache:.2f} ms/步"
              f"（每步只算 1 个 token）")
        print(f"[naive 版] 全量    : {t_naive * 1000:.1f} ms / {n_naive} 步 = {ms_naive:.2f} ms/步"
              f"（每步重算全部历史）")
        print("-" * 58)
        print(f"总耗时：cache {total_cache * 1000:.1f} ms  vs  naive {t_naive * 1000:.1f} ms"
              f"  →  {t_naive / max(1e-9, total_cache):.2f}x")
        print(f"decode 单步加速比  : {ms_naive / max(1e-9, ms_cache):.2f}x  ← KV Cache 的核心收益")
        print("=" * 58)
        return {"same": same, "t_prefill": t_prefill, "t_decode": t_decode,
                "t_naive": t_naive, "ms_cache": ms_cache, "ms_naive": ms_naive}

    # ==================================================================
    # 对话
    # ==================================================================
    def chat(self, prompt, max_new=50, temperature=0.0, top_k=None):
        ids, _ = self.tokenizer.render_conversation(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        out = self.generate_cached(list(ids), max_new=max_new,
                                   temperature=temperature, top_k=top_k)
        return self.tokenizer.decode(out)
