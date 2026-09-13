"""推理性能基准：TTFT / TPOT（KV Cache vs 全量重算）。

用法:
    python bench_inference.py --source sft --model-tag d8
    python bench_inference.py --contexts 32,64,128,192 --max-new 32 --repeats 3
"""
import argparse
import json
import statistics
import time

import torch

from nanochat.common import compute_init, autodetect_device_type, COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description="TTFT/TPOT 推理性能基准")
parser.add_argument("--source", type=str, default="sft", help="模型来源: base|sft")
parser.add_argument("--model-tag", type=str, default="d8")
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--text-path", type=str, default="sft_eval_cls.jsonl", help="取真实文本构造 prompt")
parser.add_argument("--contexts", type=str, default="32,64,128,192", help="测试的 prompt 长度列表")
parser.add_argument("--max-new", type=int, default=32, help="decode 步数")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--out", type=str, default="bench_results.md")
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps", ""])
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

cfg = model.config
kv_kwargs = {
    "num_heads": getattr(cfg, "n_kv_head", cfg.n_head),
    "head_dim": cfg.n_embd // cfg.n_head,
    "num_layers": cfg.n_layer,
}
n_params = sum(p.numel() for p in model.parameters())
print(f"模型: {args.source}/{args.model_tag}  参数量 {n_params/1e6:.1f}M  "
      f"层数 {cfg.n_layer}  KV头 {kv_kwargs['num_heads']}  head_dim {kv_kwargs['head_dim']}  "
      f"max_seq_len {cfg.sequence_len}  设备 {device}")


def make_prompt(n_tokens):
    """从真实新闻里取 n_tokens 个 token 作为 prompt"""
    with open(args.text_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ids = []
    for r in rows:
        ids.extend(tokenizer.encode(r["text"]))
        if len(ids) >= n_tokens:
            break
    ids = ([tokenizer.get_bos_token_id()] + ids)[:n_tokens]
    return ids


@torch.no_grad()
def bench_cached(tokens, max_new):
    """prefill 一次 + decode 每步只喂 1 个 token"""
    cache = KVCache(batch_size=1, seq_len=len(tokens) + max_new, device=device,
                    dtype=COMPUTE_DTYPE, **kv_kwargs)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    logits = model.forward(x, kv_cache=cache)
    ttft = time.perf_counter() - t0
    nxt = int(torch.argmax(logits[0, -1]).item())

    t0 = time.perf_counter()
    for _ in range(max_new):
        x = torch.tensor([[nxt]], dtype=torch.long, device=device)
        logits = model.forward(x, kv_cache=cache)
        nxt = int(torch.argmax(logits[0, -1]).item())
    tpot = (time.perf_counter() - t0) / max_new
    return ttft, tpot


@torch.no_grad()
def bench_naive(tokens, max_new):
    """无 cache：每步把全部历史重新喂进去算 K/V"""
    out = list(tokens)
    x = torch.tensor([out], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    logits = model.forward(x)
    ttft = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(max_new):
        out.append(int(torch.argmax(logits[0, -1]).item()))
        x = torch.tensor([out], dtype=torch.long, device=device)
        logits = model.forward(x)
    tpot = (time.perf_counter() - t0) / max_new
    return ttft, tpot


contexts = [int(c) for c in args.contexts.split(",")]
rows = []
for ctx in contexts:
    prompt = make_prompt(ctx)
    # 预热（第一次 forward 有 lazy 初始化开销）
    bench_cached(prompt[:min(16, len(prompt))], 2)

    cache_t, cache_p, naive_t, naive_p = [], [], [], []
    for _ in range(args.repeats):
        ttft, tpot = bench_cached(prompt, args.max_new)
        cache_t.append(ttft * 1000)
        cache_p.append(tpot * 1000)
        ttft, tpot = bench_naive(prompt, args.max_new)
        naive_t.append(ttft * 1000)
        naive_p.append(tpot * 1000)

    cache_t, cache_p = statistics.median(cache_t), statistics.median(cache_p)
    naive_t, naive_p = statistics.median(naive_t), statistics.median(naive_p)
    rows.append({
        "ctx": len(prompt), "cache_ttft": cache_t, "cache_tpot": cache_p,
        "naive_ttft": naive_t, "naive_tpot": naive_p,
        "speedup": naive_p / cache_p if cache_p else float("nan"),
    })
    print(f"ctx={len(prompt):>4}  TTFT {cache_t:7.1f} ms  |  "
          f"TPOT cache {cache_p:6.2f} ms  vs  naive {naive_p:6.2f} ms  →  {rows[-1]['speedup']:.2f}x")

# KV Cache 显存/内存占用
kv_bytes_per_token = 2 * kv_kwargs["num_layers"] * kv_kwargs["num_heads"] * kv_kwargs["head_dim"] * COMPUTE_DTYPE.itemsize
print(f"\nKV Cache 占用: {kv_bytes_per_token / 1024:.1f} KB/token"
      f"（seq=256 时 {kv_bytes_per_token * cfg.sequence_len / 1024 / 1024:.2f} MB/序列）")

with open(args.out, "w", encoding="utf-8") as f:
    f.write(f"# 推理性能基准（TTFT / TPOT）\n\n")
    f.write(f"- 模型：`{args.source}/{args.model_tag}`，{n_params/1e6:.1f}M 参数，"
            f"{cfg.n_layer} 层，KV 头 {kv_kwargs['num_heads']}，dtype {COMPUTE_DTYPE}\n")
    f.write(f"- 设备：{device}，decode 步数 {args.max_new}，取 {args.repeats} 次中位数\n")
    f.write(f"- KV Cache 占用：{kv_bytes_per_token/1024:.1f} KB/token，"
            f"seq={cfg.sequence_len} 时 {kv_bytes_per_token*cfg.sequence_len/1024/1024:.2f} MB/序列\n\n")
    f.write("| prompt 长度 | TTFT (ms) | TPOT cache (ms/tok) | TPOT 全量重算 (ms/tok) | decode 加速 |\n")
    f.write("|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['ctx']} | {r['cache_ttft']:.1f} | {r['cache_tpot']:.2f} | "
                f"{r['naive_tpot']:.2f} | **{r['speedup']:.2f}x** |\n")
print(f"\n结果已写入 {args.out}")
