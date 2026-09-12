# -*- coding: utf-8 -*-
"""
mini_engine.py —— 从零实现的 KV Cache 推理引擎

自己实现「prefill + decode」两阶段自回归生成：
  - prefill：prompt 的 K/V 一次算好，写进 KV Cache
  - decode ：每步【只喂新 token】，历史 K/V 从 cache 复用（不重算）

同时提供「朴素生成」（每步喂全部历史、重算 K/V）作为对照，用来验证：
  **KV Cache 不改变结果，但显著减少计算量。**

用法（在 self_nanochat 根目录下）：
    set PYTHONUTF8=1                                  # Windows 中文输出
    python mini_engine.py --model-tag d2 --prompt "你好" --max-new 20
    python mini_engine.py --model-tag d2 --compare    # cache vs naive 对比实验
"""
import os
import sys
import time
import argparse

import torch

# Windows 控制台可能是 GBK，中文输出兜底
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from nanochat.checkpoint_manager import load_model
from nanochat.engine import KVCache
from nanochat.common import COMPUTE_DTYPE


class MiniEngine:
    """从零实现的 KV Cache 推理引擎（单样本、增量解码）"""

    def __init__(self, model, tokenizer, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

        # KV Cache 的维度从模型配置取（注意 GQA：KV 头数可能小于 Q 头数）
        m = self.model.config
        self.kv_kwargs = {
            "num_heads": getattr(m, "n_kv_head", m.n_head),
            "head_dim": m.n_embd // m.n_head,
            "num_layers": m.n_layer,
        }
        self.max_seq_len = m.sequence_len          # 模型 RoPE 缓存上限
        self.stop_id = self.tokenizer.encode_special("<|assistant_end|>")

    # ------------------------------------------------------------------
    # ① 带 KV Cache 的生成（prefill + decode）
    # ------------------------------------------------------------------
    def generate(self, tokens, max_new=50, temperature=0.0, top_k=None, seed=42,
                 stop_at_end=True):
        tokens = list(tokens)
        # 不能超过模型支持的序列长度
        if len(tokens) >= self.max_seq_len:
            tokens = tokens[-(self.max_seq_len - 1):]
        cache_len = max(min(len(tokens) + max_new, self.max_seq_len), len(tokens))

        cache = KVCache(
            batch_size=1,
            seq_len=cache_len,
            device=self.device,
            dtype=COMPUTE_DTYPE,
            **self.kv_kwargs,
        )

        generator = None
        if temperature > 0:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        out = tokens
        with torch.no_grad():
            # --- prefill：prompt 一次算完，K/V 写入 cache ---
            x = torch.tensor([tokens], dtype=torch.long, device=self.device)
            logits = self.model.forward(x, kv_cache=cache)

            # --- decode：每步只喂新 token，历史 K/V 复用 ---
            for _ in range(max_new):
                if len(out) >= cache_len:
                    break
                nxt = self._sample(logits[0, -1], temperature, top_k, generator)
                out.append(nxt)
                if stop_at_end and nxt == self.stop_id:
                    break
                x = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
                logits = self.model.forward(x, kv_cache=cache)
        return out

    # ------------------------------------------------------------------
    # ② 朴素生成（无 cache：每步喂全部历史，重算 K/V）—— 对照实现
    # ------------------------------------------------------------------
    def generate_naive(self, tokens, max_new=50, temperature=0.0, top_k=None, seed=42,
                       stop_at_end=True):
        tokens = list(tokens)
        generator = None
        if temperature > 0:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        out = tokens
        with torch.no_grad():
            for _ in range(max_new):
                if len(out) >= self.max_seq_len:
                    break
                x = torch.tensor([out], dtype=torch.long, device=self.device)
                logits = self.model.forward(x)      # 无 cache → 全量重算
                nxt = self._sample(logits[0, -1], temperature, top_k, generator)
                out.append(nxt)
                if stop_at_end and nxt == self.stop_id:
                    break
        return out

    def _sample(self, logits, temperature, top_k, generator):
        """temperature<=0 → 贪心；否则按温度采样（可选 top-k）"""
        if temperature is None or temperature <= 0:
            return int(torch.argmax(logits).item())
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.clone()
            logits[logits < v[-1]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1, generator=generator).item())

    # ------------------------------------------------------------------
    # ③ 对话
    # ------------------------------------------------------------------
    def chat(self, prompt, max_new=50, temperature=0.0, top_k=None):
        ids, _ = self.tokenizer.render_conversation(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        out = self.generate(list(ids), max_new=max_new, temperature=temperature, top_k=top_k)
        return self.tokenizer.decode(out)

    # ------------------------------------------------------------------
    # ④ 对比实验：cache vs naive（结果应一致 + 速度对比）
    # ------------------------------------------------------------------
    def compare(self, tokens, max_new=50):
        """
        公平对比：贪心解码 + 强制生成满额（不考虑停止 token）
        分开统计 prefill / decode，突出 KV Cache 的真正价值：decode 步只算 1 个 token
        """
        tokens = list(tokens)
        if len(tokens) >= self.max_seq_len:
            tokens = tokens[-(self.max_seq_len - 1):]
        cache_len = max(min(len(tokens) + max_new, self.max_seq_len), len(tokens))

        # ---------------- ① cache 版：prefill + decode ----------------
        cache = KVCache(batch_size=1, seq_len=cache_len, device=self.device,
                        dtype=COMPUTE_DTYPE, **self.kv_kwargs)
        with torch.no_grad():
            x = torch.tensor([tokens], dtype=torch.long, device=self.device)
            t0 = time.time()
            logits = self.model.forward(x, kv_cache=cache)      # prefill：prompt 一次算
            t_prefill = time.time() - t0

            out_cache = list(tokens)
            t0 = time.time()
            n_cache_steps = 0
            for _ in range(max_new):
                if len(out_cache) >= cache_len:
                    break
                nxt = int(torch.argmax(logits[0, -1]).item())
                out_cache.append(nxt)
                x = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
                logits = self.model.forward(x, kv_cache=cache)  # decode：只喂 1 个 token
                n_cache_steps += 1
            t_decode = time.time() - t0

        # ---------------- ② naive 版：每步全量重算 ----------------
        with torch.no_grad():
            out_naive = list(tokens)
            t0 = time.time()
            n_naive_steps = 0
            for _ in range(max_new):
                if len(out_naive) >= self.max_seq_len:
                    break
                x = torch.tensor([out_naive], dtype=torch.long, device=self.device)
                logits = self.model.forward(x)                  # 无 cache：全量重算
                nxt = int(torch.argmax(logits[0, -1]).item())
                out_naive.append(nxt)
                n_naive_steps += 1
            t_naive = time.time() - t0

        same = out_cache == out_naive
        ms_cache = 1000 * t_decode / max(1, n_cache_steps)
        ms_naive = 1000 * t_naive / max(1, n_naive_steps)
        print("=" * 58)
        print("KV Cache 对比实验（贪心解码，强制生成满额）")
        print("=" * 58)
        print(f"prompt tokens      : {len(tokens)}")
        print(f"新生成 tokens      : {n_cache_steps}")
        print(f"结果一致           : {same}   ← 正确性：cache 不改变结果")
        print("-" * 58)
        print(f"[cache 版] prefill : {t_prefill * 1000:.1f} ms（prompt 一次算完）")
        print(f"[cache 版] decode  : {t_decode * 1000:.1f} ms / {n_cache_steps} 步 "
              f"= {ms_cache:.2f} ms/步（每步只算 1 个 token）")
        print(f"[naive 版] 全量    : {t_naive * 1000:.1f} ms / {n_naive_steps} 步 "
              f"= {ms_naive:.2f} ms/步（每步重算全部历史）")
        print("-" * 58)
        total_cache = t_prefill + t_decode
        print(f"总耗时：cache {total_cache * 1000:.1f} ms  vs  naive {t_naive * 1000:.1f} ms"
              f"  →  {t_naive / max(1e-9, total_cache):.2f}x")
        print(f"decode 单步加速比  : {ms_naive / max(1e-9, ms_cache):.2f}x  ← KV Cache 的核心收益")
        print("=" * 58)
        return {"same": same, "t_prefill": t_prefill, "t_decode": t_decode,
                "t_naive": t_naive, "ms_cache": ms_cache, "ms_naive": ms_naive}


def main():
    parser = argparse.ArgumentParser(description="MiniEngine：从零实现的 KV Cache 推理引擎")
    parser.add_argument("--model-tag", type=str, default="d2")
    parser.add_argument("--source", type=str, default="sft", choices=["sft", "base"],
                        help="加载 sft（对话微调）或 base（预训练）checkpoint")
    parser.add_argument("--prompt", type=str, default="你好")
    parser.add_argument("--max-new", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda（默认自动检测）")
    parser.add_argument("--compare", action="store_true", help="跑 cache vs naive 对比实验")
    parser.add_argument("--compare-tokens", type=int, default=32,
                        help="对比实验的 prompt 长度（重复填充到该长度，放大 cache 收益）")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MiniEngine] 加载 {args.source} 模型：model-tag={args.model_tag} device={device}")
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag)

    engine = MiniEngine(model, tokenizer, device=device)
    print(f"[MiniEngine] 模型配置：n_layer={model.config.n_layer} n_head={model.config.n_head} "
          f"n_kv_head={getattr(model.config, 'n_kv_head', model.config.n_head)} "
          f"seq_len={model.config.sequence_len}")

    if args.compare:
        ids, _ = tokenizer.render_conversation(
            {"messages": [{"role": "user", "content": args.prompt}]})
        ids = list(ids)
        # 把 prompt 重复填充到指定长度（放大 cache 的收益）
        target = max(len(ids), args.compare_tokens)
        reps = (target + len(ids) - 1) // len(ids)
        long_ids = (ids * reps)[:target]
        engine.compare(long_ids, max_new=args.max_new)
    else:
        print("-" * 56)
        print(f"用户: {args.prompt}")
        text = engine.chat(args.prompt, max_new=args.max_new,
                           temperature=args.temperature, top_k=args.top_k)
        print(f"模型: {text}")
        print("-" * 56)


if __name__ == "__main__":
    main()
