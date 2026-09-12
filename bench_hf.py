"""用同一套 TTFT/TPOT 方法测 HuggingFace 模型的推理性能（对标自研 40M 模型）。

TTFT = Time To First Token（prefill 耗时）
TPOT = Time Per Output Token（decode 阶段每个 token 的耗时）

用法:
    python bench_hf.py --model Qwen/Qwen2.5-0.5B-Instruct --contexts 32,64,128 --max-new 16
    python bench_hf.py --model Qwen/Qwen2.5-0.5B-Instruct --dtype bfloat16   # GPU 上用

和 bench_inference.py 的区别：那个测的是自己的 nanochat/MiniEngine 链路，
这个测 HuggingFace 模型，两者用同一套指标，可以直接对比。
"""
import argparse
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="HuggingFace 模型 TTFT/TPOT 基准")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--contexts", type=str, default="32,64,128")
parser.add_argument("--max-new", type=int, default=16, help="decode 步数")
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
model.eval()

cfg = model.config
n_params = sum(p.numel() for p in model.parameters())
n_kv_head = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
head_dim = cfg.hidden_size // cfg.num_attention_heads
n_layers = cfg.num_hidden_layers
print(f"模型: {args.model}")
print(f"  参数量 {n_params / 1e6:.1f}M  层数 {n_layers}  Q头 {cfg.num_attention_heads}  "
      f"KV头 {n_kv_head}  head_dim {head_dim}  dtype {args.dtype}")

text = "人工智能的未来是充满想象力的，从语言理解到多模态推理，模型正在改变我们做事的方式。" * 40
ids_all = tokenizer(text)["input_ids"]
print(f"  可用 prompt token: {len(ids_all)}")


@torch.no_grad()
def bench(n_tokens):
    """返回 (TTFT 秒, TPOT 秒/token)"""
    inp = torch.tensor([ids_all[:n_tokens]])
    t0 = time.perf_counter()
    out = model(inp, use_cache=True)          # prefill：一次算完 prompt，写入 KV cache
    ttft = time.perf_counter() - t0
    past = out.past_key_values
    nxt = int(out.logits[0, -1].argmax().item())

    t0 = time.perf_counter()
    for _ in range(args.max_new):
        out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)  # 只喂 1 个 token
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
    return ttft, (time.perf_counter() - t0) / args.max_new


rows = []
for ctx in [int(c) for c in args.contexts.split(",")]:
    ctx = min(ctx, len(ids_all))
    bench(min(8, ctx))                        # 预热
    reps = [bench(ctx) for _ in range(args.repeats)]
    ttft = statistics.median(r[0] for r in reps) * 1000
    tpot = statistics.median(r[1] for r in reps) * 1000
    rows.append((ctx, ttft, tpot))
    print(f"ctx={ctx:>5}  TTFT {ttft:8.1f} ms   TPOT {tpot:8.2f} ms/token")

itemsize = torch.tensor([], dtype=getattr(torch, args.dtype)).element_size()
kv_kb = 2 * n_layers * n_kv_head * head_dim * itemsize / 1024
print(f"\nKV Cache 占用: {kv_kb:.1f} KB/token（2 × {n_layers} 层 × {n_kv_head} KV头 × {head_dim} × {itemsize} 字节）")
print(f"对比：自研 40.7M 模型（8 层 / 4 KV 头 / head_dim 128 / fp32）= 32.0 KB/token")
print("\n| prompt 长度 | TTFT (ms) | TPOT (ms/tok) |")
print("|---|---|---|")
for ctx, ttft, tpot in rows:
    print(f"| {ctx} | {ttft:.1f} | {tpot:.2f} |")
