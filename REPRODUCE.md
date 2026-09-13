# 复现指南（REPRODUCE）

> 本仓库所有数字都在**本地 CPU**（Windows 11 + conda 环境，torch 2.5 系，无 GPU）上实测得到。
> 每个实验都给出**可复制的命令**和**预期结果**；如果对不上，先看文末「已知偏差」。

**运行前一次性设置**（Windows PowerShell）：

```powershell
$env:PYTHONUTF8="1"      # 中文编码，不加会 GBK 报错
$env:NANOCHAT_LOCAL="1"  # 本地模式：禁用 H100 专属 FP8、截断评估步数
```

**零数据冒烟测试**（约 3 秒，验证 KV Cache 与全量重算数值等价）：

```powershell
pytest tests/test_kvcache_equiv.py tests/test_engine.py tests/test_attention_fallback.py tests/test_optim.py -q
# 预期：13 passed, 14 skipped
```

---

## 一、数据准备

```powershell
# 10 类中文新闻（每类 5000 条，格式 `标签\t正文`），放到仓库根目录
#   cnews.train.txt  约 124 MB / 50,041 行
# 小样本（仅体育类，用于快速试验）
#   cnews_small.txt  约 12 MB / 5,000 行
```

## 二、八个实验的复现命令与预期结果

### 实验 1：tokenizer 词表 × 模型规模

```powershell
python -m scripts.tok_train --text-path=cnews_small.txt --max-chars=5000000 --vocab-size=5000
python -m scripts.base_train --model-tag d8 --depth 8 --max-seq-len 256 --text-path cnews_small.txt `
    --num-iterations 500 --device-batch-size 1 --total-batch-size 256 --eval-every -1 --run dummy
```

**预期**：vocab 5000 时 `你好！很高兴见到你` 切成 14 个词级 token（vocab 1000 时 50+ 字节碎片、解码乱码）；40.7M 模型 KV Cache decode 加速 3.03x（0.5M 模型仅 1.35x）。
**耗时**：分词器约 1 分钟；预训练 500 步约 14 分钟。

### 实验 2：SFT 数据规模（同一基座，只换对话数据）

```powershell
python build_sft_data.py --text-path cnews.train.txt --out local_chat_v7.jsonl `
    --cls-ratio 0.6 --sum-ratio 0.2 --identity-copies 30
python -m scripts.chat_sft --model-tag d8 --max-seq-len 256 --text-path=local_chat_v7.jsonl `
    --num-iterations 600 --device-batch-size 1 --total-batch-size 256 `
    --eval-every -1 --chatcore-every -1 --eval-tokens 2048 --run=dummy
python sft_eval.py --source sft --model-tag d8 --num-samples 1000 --show 0
```

**预期**：`local_chat_v7.jsonl` 共 31,470 条（分类 18,005 / 概括 5,967 / 续写 6,028 + 人格 1,470）；SFT 后 10 类新闻分类 **57.0%**（1000 条留出集），随机基线 10%。
**对照**：把 `--text-path` 换成 49 种对话重复 240 次的旧数据，准确率掉到 **0%**（模型退化成复读最高频答案）。
**耗时**：SFT 600 步约 23 分钟，评测 1000 条约 10 分钟。

### 实验 3：灾难性遗忘（分阶段训练）

```powershell
# 从已有对话模型续训人格（需要 --init-source sft --init-tag <上一个模型的 tag>）
python -m scripts.chat_sft --model-tag d8p --init-source sft --init-tag d8 `
    --max-seq-len 256 --text-path local_chat_identity.jsonl --num-iterations 60 `
    --device-batch-size 1 --total-batch-size 256 --eval-every -1 --chatcore-every -1 --run dummy
python sft_eval.py --source sft --model-tag d8p --num-samples 200 --show 0
```

**预期**：分类从 58.5% 崩到 **9.0%**（≈随机）；闲聊"你好"变为正确回答"你好！很高兴见到你，有什么可以帮你？"
**对照**：人格 + 分类 replay 各半 → 62~63%（保任务、人格学不会）；纯人格 + 低 LR（`--matrix-lr 0.004`）→ 48.0%。
**耗时**：每次续训 60 步约 2 分钟。

### 实验 4：推理性能 TTFT / TPOT

```powershell
python bench_inference.py --source sft --model-tag d8 --contexts 32,64,128,192 --max-new 32
```

**预期**：TPOT ≈ **14 ms/token 且与上下文长度无关**；全量重算 23.5 → 57.2 ms/token；decode 加速 **1.64x → 4.17x**；KV Cache 32.0 KB/token。
**耗时**：约 1 分钟。

### 实验 5：预训练数据扩容

```powershell
python -m scripts.base_train --model-tag d8m --depth 8 --max-seq-len 256 `
    --text-path cnews.train.txt --num-iterations 1500 `
    --device-batch-size 1 --total-batch-size 256 --eval-every 750 --run dummy
python -m scripts.chat_sft --model-tag d8m --max-seq-len 256 --text-path=local_chat_v7.jsonl `
    --num-iterations 600 --device-batch-size 1 --total-batch-size 256 `
    --eval-every -1 --chatcore-every -1 --eval-tokens 2048 --run=dummy
python sft_eval.py --source sft --model-tag d8m --num-samples 1000 --show 0
```

**预期**：基座 val bpb **1.98**（老基座未测，训练 loss 7.21 vs 6.0），但 SFT 分类 **54.5% vs 老基座 57.0%** —— 差异在噪声内（两比例检验 p≈0.26）。
**耗时**：预训练 1500 步约 50 分钟 + SFT 23 分钟。

### 实验 6：对标真实模型（Qwen2.5-0.5B）

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python bench_hf.py --model Qwen/Qwen2.5-0.5B-Instruct --contexts 32,64,128 --max-new 8
```

**预期**：494M 参数 / 24 层 / 14 Q 头 / **2 KV 头**；TTFT@128 = 539 ms；TPOT ≈ **121 ms/token**；KV Cache **24.0 KB/token**（比自研 40.7M 模型的 32.0 更小）。
**耗时**：首次需下载模型（约 1 GB），之后约 3 分钟。

### 实验 7：评测输入长度的影响

```powershell
python sft_eval.py --source sft --model-tag d8m --num-samples 200 --prefix-chars 160
```

**预期**：前缀 64 字 **59.5%** → 160 字 **21.5%**（训练数据统一用 64 字前缀，输入变长即 OOD）。
**耗时**：约 3 分钟。

### 实验 8：训练步瓶颈分析

```powershell
python profile_step.py --source sft --model-tag d8
```

**预期**：forward+backward **303 ms** vs `optimizer.step()` **1880 ms**（Muon 的 Newton-Schulz，占 86%）；梯度累积 ×8 使吞吐 **118.8 → 457.1 tok/s（3.85x）**；线程 1→8 提升 4.0x、8→14 无收益。
**耗时**：约 2 分钟。

---

## 三、已知偏差（诚实声明）

| 项 | 说明 |
|---|---|
| 硬件 | 全部为**本地 CPU**（16 GB 内存，14 线程），无 GPU；绝对耗时不可与 GPU 结果比较 |
| 随机性 | 关键结论为**单种子**结果；1000 条评测的标准误约 1.6%，200 条约 3.5%（报数字时必须带样本量） |
| 规模 | 模型 40.7M、语料约 3100 万 token —— 比主流小 3~4 个数量级，结论**只在这个区间内成立** |
| 数据 | `cnews*.txt` 不入库（`.gitignore`），需自备；HF 模型需 `HF_ENDPOINT` 镜像 |
| 环境坑 | 本地 `torch 2.13` 与 `torchvision 0.20.1` ABI 不匹配会导致**所有 HF 模型加载失败**（`operator torchvision::nms does not exist`）→ 卸载 torchvision |
