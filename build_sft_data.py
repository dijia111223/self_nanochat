"""从 cnews 语料构造 SFT 对话数据（分类 / 概括 / 续写 + 少量身份问答）。

划分：每类先留出若干条做评测集（sft_eval_cls.jsonl），其余用于构造训练对话。

用法:
    python build_sft_data.py --text-path ../llm-inference-journey/cnews.train.txt \
        --out local_chat_v6.jsonl --eval-out sft_eval_cls.jsonl --num-rows 30000
"""
import argparse
import json
import os
import random
import re

CLS_PROMPT = "下面这条新闻属于什么类别？只回答类别。\n"
SUM_PROMPT = "用一句话概括这条新闻。\n"
CONT_PROMPT = "续写这条新闻。\n"


def first_sentence(text, max_len=50):
    s = re.split(r"[。！？!?]", text, maxsplit=1)[0].strip()
    return (s[:max_len] + "。") if s else ""


def load_pairs(text_path, min_len):
    pairs = []
    with open(text_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "\t" not in line:
                continue
            label, text = line.split("\t", 1)
            label, text = label.strip(), text.strip()
            if len(text) >= min_len:
                pairs.append((label, text))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-path", type=str, required=True)
    parser.add_argument("--out", type=str, default="local_chat_v6.jsonl")
    parser.add_argument("--eval-out", type=str, default="sft_eval_cls.jsonl")
    parser.add_argument("--num-rows", type=int, default=30000)
    parser.add_argument("--eval-per-class", type=int, default=100)
    parser.add_argument("--identity-path", type=str, default="local_chat_v5.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    pairs = load_pairs(args.text_path, min_len=200)
    print(f"语料: {len(pairs):,} 条")

    by_label = {}
    for label, text in pairs:
        by_label.setdefault(label, []).append(text)
    print(f"类别: {len(by_label)} 个 -> {sorted(by_label)}")

    # 每类留出评测样本
    eval_rows, train_pairs = [], []
    for label, texts in sorted(by_label.items()):
        random.shuffle(texts)
        k = min(args.eval_per_class, len(texts) // 10)
        for t in texts[:k]:
            eval_rows.append({"label": label, "text": t})
        for t in texts[k:]:
            train_pairs.append((label, t))
    with open(args.eval_out, "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"评测集: {len(eval_rows):,} 条 -> {args.eval_out}")

    random.shuffle(train_pairs)
    rows, counts = [], {"cls": 0, "sum": 0, "cont": 0}
    for label, text in train_pairs:
        if len(rows) >= args.num_rows:
            break
        r = random.random()
        if r < 0.5:
            task, user, assistant = "cls", CLS_PROMPT + text[:64], f"{label}。"
        elif r < 0.8:
            task, user, assistant = "sum", SUM_PROMPT + text[:96], first_sentence(text)
        else:
            task, user, assistant = "cont", CONT_PROMPT + text[:48], text[48:180]
        if len(assistant.strip()) < 2:
            continue
        rows.append({"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]})
        counts[task] += 1

    if args.identity_path and os.path.exists(args.identity_path):
        seen = {}
        with open(args.identity_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    conv = json.loads(line)
                    seen.setdefault(json.dumps(conv, ensure_ascii=False), conv)
        rows.extend(seen.values())
        print(f"身份问答: {len(seen)} 条")

    random.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"任务分布: 分类 {counts['cls']:,} / 概括 {counts['sum']:,} / 续写 {counts['cont']:,}")
    print(f"写出 {len(rows):,} 条 -> {args.out}")


if __name__ == "__main__":
    main()
