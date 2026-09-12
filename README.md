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

## 从零实现的推理引擎（mini_engine.py）

不依赖 nanochat 的 `engine.py`，自己实现 **KV Cache 增量解码**（prefill + decode 两阶段）：

```bash
# 对话生成（带 KV Cache）
python mini_engine.py --model-tag d2 --prompt "你好" --max-new 20

# 对比实验：cache vs naive（无 cache，每步重算全部历史）
python mini_engine.py --model-tag d2 --compare --compare-tokens 8 --max-new 110
```

**实现要点**：
- **prefill**：prompt 的 K/V 一次算完，写入 KV Cache
- **decode**：每步**只喂 1 个新 token**，历史 K/V 从 cache 复用（不重算）
- **GQA 支持**：cache 的 KV 头数取 `n_kv_head`（可与 Q 头数不同）
- **正确性验证**：与朴素实现（每步全量重算）贪心解码结果**完全一致**

**实测（CPU，depth=2 小模型，prompt 9 + 生成 110 tokens）**：

| 指标 | 结果 |
|---|---|
| 结果一致 | ✅ True（cache 不改变结果） |
| prefill | 4.1 ms |
| cache decode | 1.57 ms/步（每步只算 1 个 token） |
| naive 每步 | 2.12 ms/步（每步重算全部历史） |
| decode 单步加速 | **1.35x** |

> 注：小模型 + 短序列时收益有限（框架固定开销占主导）；**序列越长、模型越大，KV Cache 收益越显著**（attention 计算从 O(T²) 降到 O(T)）。

## 扩展点（本项目增量）

- `nanochat/dataloader.py`：`txt_to_docs`（line_mode 按行切）+ `_text_document_batches`（本地文本数据迭代）
- `scripts/chat_sft.py`：`--text-path` + `LocalChatDataset`（本地对话数据）
- `nanochat/common.py`：`LOCAL_MODE` 双模式开关
- `local_eval.py`：bpb 评估 + 对话测试
- `mini_engine.py`：从零实现的 KV Cache 推理引擎（prefill/decode + 对比实验）

## 文档

- [SFT实战记录.md](SFT实战记录.md) —— 从零训练到对话的完整实战记录 + 踩坑

## 致谢

基于 [karpathy/nanochat](https://github.com/karpathy/nanochat) 扩展。
