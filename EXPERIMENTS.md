# 实验记录（self_nanochat）

所有实验都在同一台机器上完成：Windows + CPU（无 GPU），conda 环境 `ai`，torch 2.5.1。
基础模型：depth 8 / n_embd 512 / vocab 5000 / 40.7M 参数 / ctx 256，`COMPUTE_DTYPE=float32`。

复现统一前置：

```powershell
$env:PYTHONUTF8="1"; $env:NANOCHAT_LOCAL="1"
```

---

## 实验 1：tokenizer 词表 × 模型规模

| 配置 | vocab | 参数量 | 中文生成 | KV Cache decode 加速 |
|---|---|---|---|---|
| 初始 | 1000 | 0.5M | ❌ 乱码（字节碎片） | 1.35x |
| 提升后 | 5000 | 40.7M | ✅ 可读中文（真实新闻语言） | 3.03x |

- vocab 1000 时 `你好！很高兴见到你` 被切成 50+ 个字节碎片；vocab 5000 后 14 个词级 token，decode 完全可读
- 模型从 0.5M 到 40.7M，KV Cache 收益同步从 1.35x 提到 3.03x（模型越大，省下的计算越多）

```powershell
python -m scripts.tok_train --text-path=cnews_small.txt --max-chars=5000000 --vocab-size=5000
python -m scripts.base_train --model-tag d8 --depth 8 --max-seq-len 256 --text-path cnews_small.txt `
    --num-iterations 500 --device-batch-size 1 --total-batch-size 256 --eval-every -1 --run dummy
```

## 实验 2：SFT 数据规模（同一基座，只换对话数据）

评测：10 类新闻分类，1000 条留出集抽 200 条，贪心解码，随机基线 10%（`sft_eval.py`）。

| 对话数据 | 唯一对话数 | 步数 | val bpb | 分类准确率 | 对话表现 |
|---|---|---|---|---|---|
| 基座 d8（未 SFT） | — | — | — | 0/100 = 0% | 复读机"，分差，分差…" |
| v5（49 种 × 240 重复） | 49 | 300 | 2.9464 | 0/100 = 0% | 崩坏：问什么都答"GPU上的内存。" |
| v6（30,049 条多任务） | 30,049 | 600 | 3.2637 | 104/200 = 52.0% | 分类指令跟随正确，闲聊丢失 |
| v7（分类加权 + 人格 4.7%） | 31,470 | 600 | — | 117/200 = 58.5% | 闲聊仍丢失 |

分类别准确率（v7）：体育 90% / 财经 86% / 娱乐 82% / 游戏 69% / 房产 57% / 时政 53% / 科技 53% / 时尚 52% / 教育 38% / **家居 0%**（全被判成房产）

结论：
- **数据多样性 > 数据条数**：49 种答案重复 240 次，交叉熵最优解就是"输出最高频答案"
- bpb 不能跨数据集比较：v6 的验证集是多样的新闻对话，v5 只有 49 种固定答案（本身高度可预测）

```powershell
python build_sft_data.py --text-path cnews.train.txt --out local_chat_v7.jsonl `
    --cls-ratio 0.6 --sum-ratio 0.2 --identity-copies 30
python -m scripts.chat_sft --model-tag d8 --max-seq-len 256 --text-path local_chat_v7.jsonl `
    --num-iterations 600 --device-batch-size 1 --total-batch-size 256 --eval-every -1 --chatcore-every -1 --run dummy
python sft_eval.py --source sft --model-tag d8 --num-samples 200
```

## 实验 3：灾难性遗忘（人格 vs 任务，分阶段训练）

给 `chat_sft.py` 加了 `--init-source sft --init-tag <tag>`，可以从已有对话模型续训。

| 方案 | 分类准确率 | 闲聊"你好" |
|---|---|---|
| v7 单阶段（人格 4.7%） | 58.5% | ❌ "时尚。" |
| v7 → 纯人格续训 60 步 | **9.0%**（≈随机，崩） | ✅ "你好！很高兴见到你，有什么可以帮你？" |
| v7 → 人格 + 分类 replay 各半，60 步 | **63.0%** | ❌ "推理时重复计算。" |
| v7 → 纯人格 + 低 LR（matrix_lr 0.004），40 步 | 48.0% | ❌ 退化输出 |

- **灾难性遗忘 60 步就能发生**：纯人格训练把分类从 58.5% 打到 9.0%
- **replay 保任务、不保新能力**：混一半分类数据后任务更强（63.0%），但人格学不会（人格样本短，监督量被稀释）
- **低 LR 只减缓不解决**（48.0%）
- 真实模型的解法：LoRA / 任务模板或系统提示区分 / 更大模型与更多数据

```powershell
# 从 v7 模型续训人格（60 步）
python -m scripts.chat_sft --model-tag d8p --init-source sft --init-tag d8 `
    --text-path local_chat_identity.jsonl --num-iterations 60 `
    --device-batch-size 1 --total-batch-size 256 --eval-every -1 --chatcore-every -1 --run dummy
```

## 实验 4：推理性能 TTFT / TPOT

`bench_inference.py`，40.7M 模型，CPU，贪心解码，3 次取中位数。

| prompt 长度 | TTFT (ms) | TPOT cache (ms/tok) | TPOT 全量重算 (ms/tok) | decode 加速 |
|---|---|---|---|---|
| 32 | 20.9 | 14.31 | 23.46 | 1.64x |
| 64 | 30.1 | 14.24 | 30.46 | 2.14x |
| 128 | 50.1 | 14.12 | 45.33 | 3.21x |
| 192 | 53.7 | **13.71** | 57.20 | **4.17x** |

- **TPOT 是常数**（~14 ms/token）：decode 每步只算 1 个 token，O(1)
- **全量重算的 TPOT 线性增长**（23 → 57 ms）：每步重算全部历史，O(T)
- KV Cache 占用 32 KB/token（fp32），seq=256 时 8 MB/序列

```powershell
python bench_inference.py --source sft --model-tag d8 --contexts 32,64,128,192 --max-new 32
```

## 实验 5：预训练数据扩容

基座 d8 只训了 500 步（12.8 万 token）且语料是**单一体育类**（cnews_small.txt 5000 条全是"体育"）。
把语料换成 10 类 4.3 万条（cnews.train.txt，约 3100 万 token），步数提到 1500（38.4 万 token）：

| | 老基座 d8 | 新基座 d8m |
|---|---|---|
| 语料 | 5000 篇 / **全是体育** | 43,245 篇 / 10 类 |
| 步数 / token | 500 / 12.8 万 | 1500 / **38.4 万** |
| 训练 loss | 8.52 → 7.21 | 8.53 → **约 6.0** |
| val bpb | 未测 | **1.98** |

两个基座各自在同一份 v7 SFT 数据上训 600 步，在**同一批 1000 条留出集**（每类 100 条，标准误 ≈1.6%）上评测：

| 基座 | SFT val bpb | 分类准确率（1000 条） |
|---|---|---|
| d8（12.8 万 token / 单一体育类） | — | **570/1000 = 57.0%** |
| d8m（38.4 万 token / 10 类） | 3.0877 | **545/1000 = 54.5%** |

分类别对比（每类 100 条）：

| 类别 | d8 | d8m |
|---|---|---|
| 体育 | 93% | 83% |
| 娱乐 | 82% | **93%** |
| 家居 | 0% | 0% |
| 房产 | 43% | 52% |
| 教育 | **33%** | 3% |
| 时尚 | 47% | **89%** |
| 时政 | 69% | 56% |
| 游戏 | 71% | 59% |
| 科技 | 47% | 30% |
| 财经 | 85% | 80% |

**结论（和最初假设相反）**：
- **预训练数据 3 倍扩容没有带来可测量的收益**：57.0% vs 54.5%，差异 2.5 个点在噪声范围内（两比例检验 p≈0.26）
- 基座 bpb 确实大幅改善（训练 loss 7.2 → 6.0，val bpb 1.98），但**没有传导到下游任务**
- 对照：SFT 数据从"49 种对话重复 240 次"换成"3 万条多任务"是 **+52 个点**（0% → 52%）
- 所以在这个规模下，**判别类能力几乎完全由 SFT 监督信号决定，预训练数据量不是瓶颈**（至少在 12.8 万 → 38.4 万 token 区间）
- 类别间表现差异大且**两个模型不一致**（时尚 47% vs 89%、教育 33% vs 3%），说明 40M 模型学到的是浅层启发式而非稳定特征
- **家居类两个模型都是 0/100** —— 系统性失败。抽样 20 条看混淆去向：预测成 **时尚 9 次、财经 6 次、房产 3 次、家居 0 次**。
  看正文就明白：家居类新闻一半是"沙发/窗帘/居室/软装"（像时尚），一半是"建材下乡/行业转型/消费白皮书"（像财经），
  **类目定义本身就和时尚、财经重叠**，而评测只给模型看前 64 个字 → 这个 0% 里有一部分是任务本身的歧义，不全是模型的锅
- 生成任务（概括/续写）仍然退化（`(记者 (记者 (记者…`）：基座 bpb 的改善没有转化成生成质量，瓶颈在模型容量与训练量级

```powershell
python -m scripts.base_train --model-tag d8m --depth 8 --max-seq-len 256 `
    --text-path cnews.train.txt --num-iterations 1500 `
    --device-batch-size 1 --total-batch-size 256 --eval-every 750 --run dummy
python -m scripts.chat_sft --model-tag d8m --max-seq-len 256 --text-path local_chat_v7.jsonl `
    --num-iterations 600 --device-batch-size 1 --total-batch-size 256 `
    --eval-every -1 --chatcore-every -1 --eval-tokens 2048 --run dummy
python sft_eval.py --source sft --model-tag d8m --num-samples 1000
```


## 实验 6：对标真实模型（Qwen2.5-0.5B-Instruct）

`bench_hf.py` 用与 `bench_inference.py` **完全相同的方法**（手动 prefill + `past_key_values` 逐步 decode）测 HuggingFace 模型。

| 指标 | 自研 d8（40.7M） | Qwen2.5-0.5B-Instruct（494M） |
|---|---|---|
| 层数 / Q头 / **KV头** / head_dim | 8 / 4 / 4 / 128 | 24 / 14 / **2** / 64 |
| TTFT @ 128 token | 50.1 ms | 539.2 ms |
| TPOT | **13.7 ms/token** | 121.4 ms/token |
| **KV Cache** | 32.0 KB/token | **24.0 KB/token** |
| TPOT 随上下文变化 | 否（常数） | 否（常数） |

**关键结论**：
- **参数量大 12 倍，KV Cache 反而小 25%**：Qwen 用 GQA——14 个 Q 头只配 **2 个 KV 头**，head_dim 也只有 64。
  → 说明推理优化的核心不是"模型多大"，而是 **decode 时每 token 要搬运多少 KV 字节**（decode 是 memory-bound）
- TPOT 与参数量近似线性：121.4 / 13.7 ≈ 8.9x，而参数比 494 / 40.7 ≈ 12.1x（decode 阶段瓶颈是读权重）
- 两个模型的 TPOT **都是常数**（不随上下文增长）——反向验证了 KV Cache 方法论的正确性
- TTFT 随上下文增长（prefill 是 compute-bound）：197.6 → 539.2 ms

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python bench_hf.py --model Qwen/Qwen2.5-0.5B-Instruct --contexts 32,64,128 --max-new 8
```

> 环境坑：本地 `torch 2.13` 与 `torchvision 0.20.1` ABI 不匹配（`RuntimeError: operator torchvision::nms does not exist`），
> 而 transformers 5.15 加载任何模型都会强制 import torchvision → **本地所有 HF 模型都加载不了**。
> 卸载 torchvision 后解决（真需要时再装与 torch 匹配的版本）。

## 实验 7：评测输入长度的影响（训练/推理格式必须一致）

同一个模型（d8m，SFT 训练数据统一用 `text[:64]` 前缀），只改**评测时**给模型看的新闻前缀长度：

| 评测前缀 | 分类准确率（200 条） |
|---|---|
| 64 字（**和训练一致**） | **119/200 = 59.5%** |
| 160 字 | **43/200 = 21.5%** |

分类别崩塌情况（前缀 64 → 160）：体育 85%→0%、财经 86%→5%、时政 41%→6%、房产 65%→4%

**结论**：
- 只是把输入变长，准确率就掉 38 个点 —— 模型学到的是"**64 字前缀 → 标签**"这个映射，输入一变长立刻 OOD
- 所以 **59.5% 不是"分类能力"，而是"在 64 字前缀这个特定格式下的表现"**：报告指标必须写清输入格式，否则数字没有意义
- 想用更长上下文分类，必须**在训练数据构造时**就用长前缀（改 `build_sft_data.py --prefix-chars`），而不是评测时临时加长
- 这也解释了真实系统为什么必须有统一的 chat template：**格式不一致，模型直接崩**（对应实验 3 的人格/任务挤压，本质都是"输入格式承载了任务语义"）

```powershell
python sft_eval.py --source sft --model-tag d8m --num-samples 200 --prefix-chars 160
```

---

## 实验 8：训练步瓶颈分析与吞吐优化

`profile_step.py`（CPU，SFT d8，batch 1×256）：

| 环节 | 耗时 |
|---|---|
| forward + backward | 303.5 ms |
| **optimizer.step()（Muon 的 Newton-Schulz 迭代）** | **1879.8 ms（占 86%）** |

梯度累积（优化器开销按"步"计、不按 token 计 → 攒大 batch 几乎免费）：

| 累积 | ms/步 | tokens/步 | 吞吐 |
|---|---|---|---|
| x1 | 2155 | 256 | 118.8 tok/s |
| x2 | 2482 | 512 | 206.3 tok/s |
| x4 | 3359 | 1024 | 304.9 tok/s |
| **x8** | 4481 | 2048 | **457.1 tok/s（3.85x）** |

CPU 线程伸缩性（forward）：1 线程 382.6 ms → 8 线程 **94.6 ms（4.0x）** → 14 线程 89.5 ms（无收益）

**结论**：
- 训练在这里**不是计算受限，是优化器受限**：Muon 的 Newton-Schulz 作用在参数矩阵上，与输入 batch 大小无关 → batch=1 时它占 86%
- 所以**加大有效 batch 是免费的杠杆**：×8 累积拿到 **3.85x** 吞吐（诚实边界：这是纯计算吞吐，真实训练还要叠加数据加载/分词开销）
- 线程数 8 是甜点，14 线程无收益 → `torch.set_num_threads(8)`，把余下线程让给 dataloader

```powershell
python profile_step.py --source sft --model-tag d8
```

---

## 文献坐标（每个实验在文献里的位置）

> 用途：让工程报告变成研究材料 —— 复试/面试被问"这相对已有工作是什么位置"时能答。
> 以下是**方向坐标**（代表作），写进正式材料前请把年份/出处核一遍。

| 实验 | 对应工作 | 你的结果相对它是什么位置 |
|---|---|---|
| 1 tokenizer × 规模 | GPT-2（byte-level BPE）；Kaplan et al. 2020 / Hoffmann et al. 2022（Chinchilla 缩放律） | 在极小规模上复现了"词表决定中文表达力"和"参数/数据共同决定上限" |
| 2 SFT 数据规模 | InstructGPT（Ouyang et al. 2022）；**LIMA**（Zhou et al. 2023，"Less Is More for Alignment"）；Self-Instruct（Wang et al. 2022） | LIMA 说"少量高质量就够"；你的**反向极端**（49 种 × 重复 240 次）直接崩坏 → 补上"**重复 ≠ 数量、多样性才是关键**"这一面 |
| 3 灾难性遗忘 | EWC（Kirkpatrick et al. 2017）；Experience Replay（Rolnick et al. 2019）；LoRA（Hu et al. 2021）；Luo et al. 2023（LLM 持续微调的遗忘实证） | 在 40M 规模**量化**了"60 步即可崩到随机水平"，并验证 replay 保任务、低 LR 只减缓 —— 小模型上的可复现证据 |
| 4 TTFT/TPOT | vLLM / **PagedAttention**（Kwon et al. 2023）；FlashAttention（Dao et al. 2022）；Orca（Yu et al. 2022，continuous batching） | 自研引擎复现了"decode O(1) vs 全量 O(T)"，指标口径与 serving 领域一致（可直接对标） |
| 5 预训练扩容 | Chinchilla（Hoffmann et al. 2022）；**TinyStories**（Eldan & Li 2023）；Emergent Abilities（Wei et al. 2022） | 你的结论（数据量 ×3 无可测量收益、SFT 主导）是缩放律在**远未饱和区**的边界条件/反例 |
| 6 对标 Qwen2.5 | MQA（Shazeer 2019）；**GQA**（Ainslie et al. 2023）；Qwen2.5 技术报告；Roofline（Williams et al. 2009） | 实测说明"KV Cache 大小由架构（KV 头数 × head_dim）决定，而非参数量"，与 GQA 的设计动机一致 |
| 7 输入长度/格式 | 提示敏感性：Lu et al. 2022（prompt 顺序/格式敏感性）；chat template 工程实践 | 量化了"格式不一致直接崩 38 个点"，是提示敏感性在**训练格式**层面的极端案例 |
| 8 训练瓶颈 | Muon（Jordan et al. 2024，技术报告）；梯度累积是标准工程手段 | 指出"小 batch + 重优化器（NS 迭代）"时瓶颈从计算转到优化器，**batch 成为免费杠杆** |

**建议阅读顺序**（与项目一一对应）：
PagedAttention → FlashAttention → GQA → LIMA → Chinchilla / TinyStories → EWC / Replay → Muon

---

## 踩坑汇总

1. **`--max-seq-len` 默认是 2048，模型却是 256** → `AssertionError: total_batch_size (256) must be a multiple of 2048`。base_train 要显式传 `--max-seq-len 256`；chat_sft 续训时已在脚本里加兜底（超过模型 `config.sequence_len` 自动下调）
2. **`--eval-tokens` 默认 40×524288 ≈ 2100 万 token** → CPU 上 `eval_steps` 8.2 万步、约 45 小时。看起来像"卡死"，实际是评估预算爆炸；LOCAL_MODE 下已截断为 8 步
3. **日志块缓冲**：Python stdout 重定向到文件是 8KB 块缓冲，日志比真实进度落后约 38 步。"日志不动 ≠ 进程不动"
4. **"卡死"先抓栈**：`py-spy dump --pid <pid>` 一条命令定位到进程卡在 `evaluate_bpb`，胜过几小时猜测
5. **逐 token 解码打碎汉字**：`tokenizer.decode([token])` 会出现 `陈水��政`，要解码整段只打印新增部分
6. **`gpt.py` 的 RoPE 表按 `config.sequence_len` 建**，输入超长会崩（`Sequence length grew beyond the rotary embeddings cache`）
7. **数据入 git**：124MB 语料不能提交，先写 `.gitignore`
