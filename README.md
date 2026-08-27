# self_nanochat —— 本地文件 LLM 训练器

> 基于 [nanochat](https://github.com/karpathy/nanochat)（karpathy）扩展：**支持本地文件训练**（不依赖云端数据集），提供"本地/云端"双模式。
> 从零跑通完整 LLM 训练流程：分词 → 预训练 → SFT（对话微调）→ 评估 → 对话。

## 特性

- **本地数据训练**：预训练读本地 txt（dataloader 扩展），SFT 读本地对话 jsonl/txt（LocalChatDataset）
- **双模式开关**：`NANOCHAT_LOCAL=1` 本地模式（禁用 H100 专属 FP8）；默认云端模式（原 nanochat）
- **全流程**：分词（tok_train）→ 预训练（base_train）→ 对话微调（chat_sft）→ 评估（local_eval）→ 对话
- **单机可跑**：CPU/小 GPU 均可（无 cl.exe 时跳过 torch.compile）

## 快速开始（本地模式）

```bash
# 1. 环境
pip install rustbpe tiktoken pyarrow wandb torch
export NANOCHAT_LOCAL=1       # Windows: $env:NANOCHAT_LOCAL="1"
export PYTHONUTF8=1           # Windows 中文编码

# 2. 训练分词器（本地文本）
python -m scripts.tok_train --max-chars=100000 --vocab-size=1000

# 3. 预训练（本地 txt）
python -m scripts.base_train --depth=2 --max-seq-len=128 --device-batch-size=1 \
  --total-batch-size=128 --num-iterations=500 --run=dummy --text-path=your_news.txt \
  --window-pattern L --eval-every -1 --core-metric-every -1 --sample-every -1 --save-every -1

# 4. SFT 对话微调（本地对话 jsonl，OpenAI 格式）
python -m scripts.chat_sft --model-tag d2 --num-iterations=200 --device-batch-size=1 \
  --total-batch-size=128 --run=dummy --text-path=your_chat.jsonl --eval-every -1 --chatcore-every -1

# 5. 评估（bpb + 对话测试）
python local_eval.py --model-tag d2 --text-path=your_news.txt

# 6. 对话
python -m scripts.chat_cli --model-tag d2
```

## 数据格式

| 阶段 | 格式 | 示例 |
|---|---|---|
| 预训练 | txt（每行一段，可选 `标签\t正文`） | `体育\t新闻内容...` |
| SFT | jsonl（OpenAI messages 格式） | `{"messages":[{"role":"user","content":"你好"},{"role":"assistant","content":"你好！"}]}` |
| SFT（txt 备选） | txt（`用户：/助手：` 空行分段） | `用户：你好\n助手：你好！` |

## 双模式说明

| 模式 | 设置 | 数据 | FP8 |
|---|---|---|---|
| **本地** | `NANOCHAT_LOCAL=1` | 本地 txt/jsonl | 禁用 |
| **云端** | 默认（不设） | HF parquet / 任务数据集 | 可用（需 H100） |

## 扩展点（本项目增量）

- `nanochat/dataloader.py`：`txt_to_docs`（line_mode 按行切）+ `_text_document_batches`（本地文本数据迭代）
- `scripts/chat_sft.py`：`--text-path` + `LocalChatDataset`（本地对话数据）
- `nanochat/common.py`：`LOCAL_MODE` 双模式开关
- `local_eval.py`：bpb 评估 + 对话测试

## 文档

- [SFT实战记录.md](SFT实战记录.md) —— 从零训练到对话的完整实战记录 + 踩坑

## 致谢

基于 [karpathy/nanochat](https://github.com/karpathy/nanochat) 扩展。
