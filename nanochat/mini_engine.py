# -*- coding: utf-8 -*-
"""
KV Cache 推理引擎，接口与 nanochat.engine.Engine 兼容，可直接替换。

    from nanochat.mini_engine import MiniEngine
    engine = MiniEngine(model, tokenizer)
    for token_column, token_masks in engine.generate(tokens, num_samples=1, max_tokens=256):
        ...

generate / generate_batch 与 Engine 签名一致；另有 generate_cached（prefill + decode）、
generate_naive（每步重算全部历史，作对照）、compare（正确性校验 + 耗时对比）、chat。
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

    def generate(self, tokens, num_samples=1, max_tokens=256, temperature=1.0, top_k=None, seed=42):
        """
        流式生成，每步 yield (token_column, token_masks)
            token_column: list[int]，长度 = num_samples，每个样本当前步的 token
            token_masks : list[int]，长度 = num_samples，1 表示生成部分
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

    def compare(self, tokens, max_new=50):
        """
        公平对比：贪心解码 + 强制生成满额（忽略停止 token）
        分开统计 prefill / decode，突出 KV Cache 的真正价值（decode 每步只算 1 个 token）
        """
        tokens = list(tokens)
        if len(tokens) >= self.max_seq_len:
            tokens = tokens[-(self.max_seq_len - 1):]
        cache_len = max(min(len(tokens) + max_new, self.max_seq_len), len(tokens))

        # cache 版：prefill + decode
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

        # naive 版：每步全量重算
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
        print(f"结果一致           : {same}   （正确性：cache 不改变结果）")
        print("-" * 58)
        print(f"[cache 版] prefill : {t_prefill * 1000:.1f} ms（prompt 一次算完）")
        print(f"[cache 版] decode  : {t_decode * 1000:.1f} ms / {n_steps} 步 = {ms_cache:.2f} ms/步"
              f"（每步只算 1 个 token）")
        print(f"[naive 版] 全量    : {t_naive * 1000:.1f} ms / {n_naive} 步 = {ms_naive:.2f} ms/步"
              f"（每步重算全部历史）")
        print("-" * 58)
        print(f"总耗时：cache {total_cache * 1000:.1f} ms  vs  naive {t_naive * 1000:.1f} ms"
              f"  →  {t_naive / max(1e-9, total_cache):.2f}x")
        print(f"decode 单步加速比  : {ms_naive / max(1e-9, ms_cache):.2f}x  （KV Cache 的核心收益）")
        print("=" * 58)
        return {"same": same, "t_prefill": t_prefill, "t_decode": t_decode,
                "t_naive": t_naive, "ms_cache": ms_cache, "ms_naive": ms_naive}

    def chat(self, prompt, max_new=50, temperature=0.0, top_k=None):
        ids, _ = self.tokenizer.render_conversation(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        out = self.generate_cached(list(ids), max_new=max_new,
                                   temperature=temperature, top_k=top_k)
        return self.tokenizer.decode(out)
