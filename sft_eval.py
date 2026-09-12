"""SFT 效果评测：新闻分类准确率（greedy 生成，只回答类别）。

用法:
    python sft_eval.py --source base --model-tag d8          # 基座基线
    python sft_eval.py --source sft  --model-tag d8          # SFT 后
"""
import argparse
import json
import random

from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

LABELS = ["体育", "娱乐", "家居", "房产", "教育", "时尚", "时政", "游戏", "科技", "财经"]
PROMPT = "下面这条新闻属于什么类别？只回答类别。\n"

parser = argparse.ArgumentParser(description="SFT 分类准确率评测")
parser.add_argument("--source", type=str, default="sft", help="模型来源: base|sft")
parser.add_argument("--model-tag", type=str, default="d8")
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--eval-path", type=str, default="sft_eval_cls.jsonl")
parser.add_argument("--num-samples", type=int, default=200)
parser.add_argument("--prefix-chars", type=int, default=64, help="给模型看的新闻前缀字数（训练时用 64）")
parser.add_argument("--max-tokens", type=int, default=8)
parser.add_argument("--show", type=int, default=6, help="打印前 N 条明细")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps", ""])
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
engine = Engine(model, tokenizer)

bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")

with open(args.eval_path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]
random.seed(args.seed)
random.shuffle(rows)
rows = rows[:args.num_samples]

correct = 0
shown = 0
per_class = {lab: [0, 0] for lab in LABELS}
for r in rows:
    tokens = [bos, user_start] + tokenizer.encode(PROMPT + r["text"][:args.prefix_chars]) + [user_end, assistant_start]
    out = []
    for token_column, token_masks in engine.generate(
        tokens, num_samples=1, max_tokens=args.max_tokens, temperature=0.0, top_k=1
    ):
        tok = token_column[0]
        if tok == assistant_end:
            break
        out.append(tok)
    text = tokenizer.decode(out).strip()
    pred = next((lab for lab in LABELS if lab in text), None)
    ok = pred == r["label"]
    correct += int(ok)
    per_class[r["label"]][1] += 1
    per_class[r["label"]][0] += int(ok)
    if shown < args.show:
        print(f"  标注={r['label']} 预测={pred} {'✓' if ok else '✗'} | 输出={text[:30]!r} | 正文={r['text'][:30]}")
        shown += 1

total = len(rows)
print(f"分类准确率: {correct}/{total} = {correct / max(1, total):.1%}  (10 类随机基线 10%)")
print("分类别准确率:")
for lab in LABELS:
    c, t = per_class[lab]
    if t:
        print(f"  {lab}: {c:>3}/{t:<3} = {c / t:.0%}")
