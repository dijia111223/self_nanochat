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

## 从零实现的推理引擎（nanochat/mini_engine.py）

与框架自带 `nanochat.engine.Engine` **接口兼容**（`generate` / `generate_batch` 签名一致），
可直接替换使用，也支持一键切换：

```bash
# 用自己的引擎对话（--engine mini）
python -m scripts.chat_cli --engine mini --model-tag d8

# 独立入口：对话 / 对比实验
python -m scripts.mini_engine --source base --model-tag d8 --prompt "人工智能" --max-new 30
python -m scripts.mini_engine --source base --model-tag d8 --compare --compare-tokens 96 --max-new 32
```

**实现要点**：
- **prefill**：prompt 的 K/V 一次算完，写入 KV Cache
- **decode**：每步**只喂 1 个新 token**，历史 K/V 从 cache 复用（不重算）
- **GQA 支持**：cache 的 KV 头数取 `n_kv_head`（可与 Q 头数不同）
- **batch 采样**：`generate` 支持 `num_samples`，流式 yield `(token_column, token_masks)`
- **正确性验证**：与朴素实现（每步全量重算）贪心解码**逐 token 一致**

**实测（CPU，贪心解码）**：

| 模型 | prompt + 生成 | 结果一致 | prefill | cache decode | naive 每步 | **decode 加速** |
|---|---|---|---|---|---|---|
| d2（0.5M 参数） | 9 + 110 | ✅ True | 4.1 ms | 1.57 ms/步 | 2.12 ms/步 | 1.35x |
| **d8（40.7M 参数）** | 96 + 32 | ✅ True | 26.0 ms | 10.35 ms/步 | 31.39 ms/步 | **3.03x** |

> **结论**：KV Cache 不改变结果（正确性验证通过），但显著减少 decode 计算；且**模型越大、序列越长，收益越显著**（0.5M → 40.7M 参数：1.35x → 3.03x；attention 计算从 O(T²) 降到 O(T)）。

## 扩展点（本项目增量）

- `nanochat/dataloader.py`：`txt_to_docs`（line_mode 按行切）+ `_text_document_batches`（本地文本数据迭代）
- `scripts/chat_sft.py`：`--text-path` + `LocalChatDataset`（本地对话数据）
- `scripts/tok_train.py`：`--text-path`（本地文本训 tokenizer）
- `nanochat/common.py`：`LOCAL_MODE` 双模式开关
- `nanochat/mini_engine.py`：从零实现的 KV Cache 推理引擎（接口兼容 + prefill/decode + 对比实验）
- `scripts/chat_cli.py`：`--engine nanochat|mini` 引擎切换
- `local_eval.py`：bpb 评估 + 对话测试
- `build_sft_data.py`：从 cnews 语料派生多任务对话数据（分类/概括/续写）
- `sft_eval.py`：SFT 效果评测（新闻分类准确率）

## 质量提升实验

### 1. tokenizer / 模型规模

| 配置 | vocab | 参数量 | 中文生成 | decode 加速 |
|---|---|---|---|---|
| 初始 | 1000 | 0.5M | ❌ 乱码（字节碎片） | 1.35x |
| **提升后** | **5000** | **40.7M** | ✅ **可读中文**（真实新闻语言） | **3.03x** |

- **tokenizer vocab 1000 → 5000**：中文从"字节级碎片"变为"字/词级"切分（`你好！很高兴见到你` 从 50+ 碎片 → 14 个词级 token），decode 完全可读
- **模型 depth 2 → 8**（0.5M → 40.7M 参数）：可学习真实的新闻语言模式；KV Cache 收益同时从 1.35x 提升到 3.03x

### 2. SFT 数据规模（同一基座 d8，只换对话数据）

评测方式：10 类新闻分类，`sft_eval.py` 从 1000 条留出集抽 200 条，greedy 解码，只回答类别（随机基线 10%）。

| 对话数据 | 唯一对话数 | 步数 | val bpb | 分类准确率 | 对话表现 |
|---|---|---|---|---|---|
| 基座 d8（未 SFT） | — | — | — | **0/100 = 0%** | 复读机："，分差，分差，分差…" |
| v5（49 种对话 × 240 重复） | 49 | 300 | 2.9464 | **0/100 = 0%** | 崩坏：问什么都答"GPU上的内存。" |
| **v6（30,049 条多任务对话）** | **30,049** | 600 | 3.2637 | **104/200 = 52.0%** | 分类指令跟随正确（"时政。"），闲聊人格丢失 |

结论：
- **数据多样性 > 数据条数**：49 种对话重复 1 万次，模型只会背最高频答案
- 换成 10 类分类/概括/续写任务后，40M 模型能学会按内容分类（52% vs 随机 10%），并学会"只回答类别"的格式
- 人格对话被淹没（49 / 30,049 = 0.16%）→ 闲聊能力要单独配比或在任务训练后再训
- v6 的 bpb 反而更高（3.26 > 2.95）：v5 的验证集只有 49 种固定答案，本身高度可预测，**bpb 不是跨数据集可比的指标**
- 概括/续写仍然退化（基座只见过 12 万 token 新闻，语言能力本身很弱）

## 文档

- [SFT实战记录.md](SFT实战记录.md) —— 从零训练到对话的完整实战记录 + 踩坑

## 致谢

基于 [karpathy/nanochat](https://github.com/karpathy/nanochat) 扩展。
