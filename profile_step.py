"""用 torch.profiler 定位训练/推理热点（CPU 本地模式）。

用法:
    python profile_step.py --model-tag d8 --source sft
"""
import argparse
import time

import torch
import torch.nn.functional as F

from nanochat.common import compute_init, autodetect_device_type, COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description="训练/推理瓶颈分析")
parser.add_argument("--source", type=str, default="sft")
parser.add_argument("--model-tag", type=str, default="d8")
parser.add_argument("--seq-len", type=int, default=256)
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps", ""])
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag)
cfg = model.config
V = cfg.vocab_size

print(f"模型 {args.source}/{args.model_tag}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
      f"| 线程数 {torch.get_num_threads()} | 设备 {device}")


def top_ops(prof, n=10):
    rows = []
    for e in prof.key_averages():
        if e.self_cpu_time_total > 0:
            rows.append((e.self_cpu_time_total / 1000.0, e.key[:60], e.count))
    rows.sort(reverse=True)
    return rows[:n]


x = torch.randint(0, V, (1, args.seq_len), device=device)
y = torch.randint(0, V, (1, args.seq_len), device=device)

# ---------- 1) 训练单步：forward + backward ----------
model.train()
for _ in range(2):
    loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
    loss.backward()
    model.zero_grad(set_to_none=True)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    for _ in range(args.repeats):
        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
print(f"\n=== 训练单步热点（forward+backward, batch 1x{args.seq_len}, {args.repeats} 步）===")
total_train = sum(r[0] for r in top_ops(prof, 999))
for ms, name, cnt in top_ops(prof):
    print(f"  {ms:8.1f} ms  {cnt:5d}x  {name}")
print(f"  （前十项占全部 CPU 时间 {100*sum(r[0] for r in top_ops(prof))/max(1e-9,total_train):.0f}%）")

# ---------- 2) 推理 decode 单步：1 token + KV Cache ----------
model.eval()
kv = {"num_heads": getattr(cfg, "n_kv_head", cfg.n_head),
      "head_dim": cfg.n_embd // cfg.n_head, "num_layers": cfg.n_layer}
cache = KVCache(batch_size=1, seq_len=args.seq_len, device=device, dtype=COMPUTE_DTYPE, **kv)
with torch.no_grad():
    model.forward(x, kv_cache=cache)          # prefill 预热
    one = torch.randint(0, V, (1, 1), device=device)
    for _ in range(3):
        model.forward(one, kv_cache=cache)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof2:
        for _ in range(20):
            model.forward(one, kv_cache=cache)
    # 计时（不用 profiler，避免开销）
    t0 = time.perf_counter()
    for _ in range(20):
        model.forward(one, kv_cache=cache)
    dec_ms = (time.perf_counter() - t0) / 20 * 1000

print(f"\n=== 推理 decode 单步热点（1 token + KV Cache，20 步）===")
print(f"  实测 {dec_ms:.2f} ms / 步")
for ms, name, cnt in top_ops(prof2, 8):
    print(f"  {ms:8.1f} ms  {cnt:5d}x  {name}")

# ---------- 4) 一个完整训练步拆解：forward+backward vs optimizer ----------
model.train()
try:
    optim = model.setup_optimizer(unembedding_lr=0.004, embedding_lr=0.3, matrix_lr=0.02, weight_decay=0.0)
except TypeError:
    optim = model.setup_optimizer()
loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
loss.backward()
optim.step()
optim.zero_grad(set_to_none=True)

n = 5
t0 = time.perf_counter()
for _ in range(n):
    loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
    loss.backward()
fb_ms = (time.perf_counter() - t0) / n * 1000

opt_times = []
for _ in range(n):
    loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
    loss.backward()
    t0 = time.perf_counter()
    optim.step()
    opt_times.append((time.perf_counter() - t0) * 1000)
    optim.zero_grad(set_to_none=True)
opt_ms = sum(opt_times) / len(opt_times)

print(f"\n=== 完整训练步拆解（batch 1x{args.seq_len}，{n} 次平均）===")
print(f"  forward+backward : {fb_ms:7.1f} ms")
print(f"  optimizer.step() : {opt_ms:7.1f} ms   ← Muon 的 Newton-Schulz 迭代在这里")
print(f"  两者合计          : {fb_ms + opt_ms:7.1f} ms")
print(f"  （chat_sft 实测一步约 2000 ms，差额 = 数据加载/分词 + 日志 + 其他开销）")

# ---------- 5) 梯度累积的实际收益（优化器开销与 batch 无关 → 攒大 batch 几乎免费）----------
print(f"\n=== 梯度累积：每步 token 数 vs 吞吐（优化器开销按步计，不按 token 计）===")
for accum in [1, 2, 4, 8]:
    for _ in range(2):                      # 预热
        for _ in range(accum):
            (F.cross_entropy(model(x).view(-1, V), y.view(-1)) / accum).backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    steps = 3
    for _ in range(steps):
        for _ in range(accum):
            (F.cross_entropy(model(x).view(-1, V), y.view(-1)) / accum).backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
    dt = (time.perf_counter() - t0) / steps
    toks = accum * args.seq_len
    print(f"  累积 x{accum}: {dt*1000:7.1f} ms/步  {toks:5d} tokens/步  →  {toks/dt:7.1f} tok/s")

print(f"\n=== CPU 线程伸缩性（forward, batch 1x{args.seq_len}）===")
orig = torch.get_num_threads()
for nt in [1, 2, 4, 8, orig]:
    if nt > orig:
        continue
    torch.set_num_threads(nt)
    with torch.no_grad():
        for _ in range(2):
            model(x)
        t0 = time.perf_counter()
        for _ in range(5):
            model(x)
        ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"  threads={nt:2d}  forward {ms:8.1f} ms")
torch.set_num_threads(orig)
