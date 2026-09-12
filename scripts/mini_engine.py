# -*- coding: utf-8 -*-
"""
scripts/mini_engine.py —— MiniEngine 的命令行入口

用法（在 self_nanochat 根目录）：
    set PYTHONUTF8=1                                       # Windows 中文输出
    python -m scripts.mini_engine --model-tag d8 --prompt "你好" --max-new 20
    python -m scripts.mini_engine --model-tag d8 --compare --compare-tokens 96 --max-new 32

说明：
    MiniEngine 与 nanochat 自带 Engine 接口兼容，
    也可用 `python -m scripts.chat_cli --engine mini` 直接对话。
"""
import os
import sys
import argparse

import torch

# Windows 控制台可能是 GBK，中文输出兜底
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from nanochat.checkpoint_manager import load_model
from nanochat.mini_engine import MiniEngine


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
    cfg = model.config
    print(f"[MiniEngine] 模型配置：n_layer={cfg.n_layer} n_head={cfg.n_head} "
          f"n_kv_head={getattr(cfg, 'n_kv_head', cfg.n_head)} "
          f"n_embd={cfg.n_embd} seq_len={cfg.sequence_len} vocab={cfg.vocab_size}")

    if args.compare:
        ids, _ = tokenizer.render_conversation(
            {"messages": [{"role": "user", "content": args.prompt}]})
        ids = list(ids)
        target = max(len(ids), args.compare_tokens)
        reps = (target + len(ids) - 1) // len(ids)
        engine.compare((ids * reps)[:target], max_new=args.max_new)
    else:
        print("-" * 56)
        print(f"用户: {args.prompt}")
        print(f"模型: {engine.chat(args.prompt, max_new=args.max_new, temperature=args.temperature)}")
        print("-" * 56)


if __name__ == "__main__":
    main()
