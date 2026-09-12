# 从零训练到能对话：self_nanochat 实战记录（2026.8.24）

> 目标：在单机/GPU 上用 self_nanochat 训练一个"能对话"的模型，验证完整流程：
> 数据准备 → 预训练 → SFT（对话微调）→ 对话测试。
> 成果：✅ 全链路打通，模型能回应"你好"→"你好！很高兴见到你"。

---

## 一、环境

- **本地**：Windows + conda ai 环境（CPU，torch 2.5.1）
- **GPU**：AutoDL 租卡（RTX 3080Ti 12GB，PyTorch 镜像，~2 元/小时）
- **代码**：self_nanochat（nanochat fork + 本地文本训练扩展）

## 二、流程与成果

### 1. 数据准备
- **预训练数据**：cnews 中文新闻（cnews.train.txt 130MB / cnews.val.txt 11.7MB，格式 `标签\t正文`）
- **SFT 数据**：自造中文对话（chinese_chat.jsonl，OpenAI 格式 `{"messages": [{"role":"user","content":...},{"role":"assistant","content":...}]}`）

### 2. 预训练（base）
- 本地 CPU：depth 2→4，bpb 6.5 → 3.27（小数据/CPU，质量有限）
- **AutoDL GPU**：depth 8，130MB 新闻，5000 步，~2.5 分钟（GPU 比 CPU 快 40 倍）
  - 结果：train bpb 0.65 / val bpb 4.34（**过拟合**——数据对模型来说太小）

### 3. SFT（对话微调）—— 我的增量
- **扩展 `scripts/chat_sft.py` 支持本地对话数据**：
  - 加 `--text-path` 参数
  - 新增 `LocalChatDataset`（读 jsonl，OpenAI 对话格式）
  - `if args.text_path is not None:` 分支：用本地对话数据，否则用自带任务数据
- **训练**：`python -m scripts.chat_sft --model-tag d8 --num-iterations=500 --text-path=chinese_chat.jsonl ...`
  - loss 正常下降（非 NaN），模型保存到 `chatsft_checkpoints/d8/`

### 4. 对话测试
```
Q: 你好
A: <|bos|><|user_start|>你好<|user_end|>你好！很高兴见到你，有什么可以帮...
```
✅ **模型会回应了**（虽然只背了训练数据里的答案——数据少 = 背答案）

## 三、踩坑记录（重要）

1. **torch.compile 需要 cl.exe**（Windows CPU）→ 注释掉所有 torch.compile（optim.py 的 @torch.compile + base_train 的 model = torch.compile）
2. **训练"卡死"**：
   - 评估环节（base_eval 下载 eval_bundle）→ 训练时 `--eval-every -1 --core-metric-every -1 --sample-every -1`
   - **113 步卡死**：数据太少（1000 条 ≈ 101K tokens）训练步数超过数据量 → refill_buffer 卡住 → **扩充数据（2 万条）解决**
3. **小数据背答案**：1000 条对话，模型 loss 0.0002（背住了）——问没见过的会乱回（无泛化）。**数据决定上限**（学过的：数据多 → 学规律 → 泛化）
4. **SFT 用英文任务数据（SmolTalk/MMLU/GSM8K）loss=nan**：语言不匹配 + 数值问题 → **改用本地中文对话数据**（自造）解决
5. **AutoDL 环境**：缺 wandb/rustbpe/tiktoken/pyarrow → 一次装齐；tokenizer 要重新训练（不跨机器共享）

## 四、简历增量（这段改动怎么描述）

> **扩展 nanochat 对话微调支持本地数据**：为 chat_sft 增加 `--text-path` 参数与 LocalChatDataset（OpenAI 对话格式 jsonl），支持自定义中文对话数据 SFT；完整跑通"新闻预训练 → 中文对话 SFT → 模型对话"全链路（AutoDL GPU）。

## 五、评估块（8.25 新增）

### local_eval.py（评估脚本）
- **bpb 评估**：`bpb = loss / ln(2)`（每字节比特数，越低越好）
- **对话测试**：对问题生成回复，观察质量
- 用法：`python local_eval.py --model-tag d2 --text-path=local_news.txt`
- 踩坑：RoPE 序列上限（d2=1280）→ 输入要截断 `model.config.sequence_len`

### 8.25 评估结果
- SFT 后新闻 bpb = 12.91（vs 预训练 3.27）——**灾难性遗忘**（对话微调覆盖新闻能力）
- 对话测试：乱码（vocab=1000 太小，中文拆成字节 token）

## 六、8.25 学习点总结

1. **bpb 评估**：loss/ln2，数值含义（bpb 3.89≈15 选 1；12.9≈7600 选 1）
2. **灾难性遗忘**（亲手观察）：SFT 后预训练能力被覆盖（新闻 bpb 3.27→12.91）——真实模型用 LoRA/混合训练/重放缓解
3. **tokenizer vocab 影响表达力**：1000 对中文太少（拆字节 → 乱码）；真实模型 15 万+
4. **模型规模是瓶颈**：depth 2 学不好 25 种对话区分——数据决定上限，但模型决定能否到上限
5. **数据耗尽卡死**：训练步数×batch > 数据量 → 卡；训练前估算数据量
6. **Windows 编码**：PYTHONUTF8=1 解决 GBK/UTF-8 冲突
7. **RoPE 上限**：输入超 model.config.sequence_len 会崩，需截断
8. **本地 vs 云端环境**：CPU 注释 torch.compile（无 cl.exe）；GPU 能编译

## 七、下一步（待做）

1. **更多对话数据**（几百~几千条不同问答）→ 模型从"背答案"到"泛化"
2. **更大模型**（depth 12+ + vocab 5000+）→ 中文表达力 + 对话区分
3. **本地 4060**：装 CUDA torch（网络问题待解决）→ 本地 GPU 训练省线
4. **chat_cli 对话接口**（SFT 模型 + 交互对话）
5. **重构双模式**（本地/云端分离，LOCAL_MODE 开关）

---

## 八、9.12 复盘：两个真 bug + 一个数据结论

### 1. "卡在 260 步"是假象（根因：评估预算 × 日志缓冲）

现象：SFT 跑到 step 260/300 后日志不再更新，CPU 一直满载，两次重跑都停在同一步、loss 数值一模一样。

排查过程：
1. `py-spy dump --pid <pid>` 抓栈 —— 进程不在数据加载里，而是卡在 **`evaluate_bpb`**（前向计算中）
2. 算预算：`--eval-tokens` 默认 `40*524288 ≈ 2100 万` token，`eval_steps = 2100万/(1*256) = 81920` 个 batch
3. CPU 上 ~2s/batch → **约 45 小时**，看起来就是死机
4. 日志"停在 260 步"是第二个假象：Python stdout 重定向到文件时是 **8KB 块缓冲**，日志比真实进度落后约 38 步；它其实早跑完 300 步了

修复：`LOCAL_MODE` 下把 `eval_steps` 截断到 8（`chat_sft.py` + `base_train.py`）；`base_train` 原本因为 `if args.eval_every > 0 and (...)` 把最终评估一起关掉了才没暴露这个问题。

教训：
- **"卡死"先抓栈，别猜**（py-spy 一条命令，胜过几小时假设）
- 日志有缓冲，**日志不动 ≠ 进程不动**
- 默认参数是按 GPU 集群定的，搬到 CPU 上要先算量级

### 2. SFT 数据多样性实验（同一个基座 d8）

| 对话数据 | 唯一对话 | 步数 | 分类准确率 | 对话表现 |
|---|---|---|---|---|
| 无（基座） | — | — | 0/100 = 0% | 复读机"，分差，分差" |
| 49 种 × 240 重复 | 49 | 300 | 0/100 = 0% | 问什么都答"GPU上的内存。" |
| 30,049 条多任务 | 30,049 | 600 | **104/200 = 52%** | 分类指令跟随正确，闲聊丢失 |

- 崩坏原因：49 种答案被重复 240 次 → 交叉熵最优解就是"无论问什么都输出最高频答案"
- 数据由 `build_sft_data.py` 从 cnews 派生（分类 14,991 / 概括 8,981 / 续写 6,028），1000 条留出做评测
- **指标要说清可比性**：v6 的 val bpb（3.26）反而比 v5（2.95）高，但 v5 的验证集只有 49 种固定答案、本身高度可预测 —— bpb 不能跨数据集比较

### 3. 还没解决的

- 概括/续写退化（`行行行行…`）：基座只训了 12.8 万 token，语言能力本身不足
- 闲聊人格丢失：身份对话只占 0.16%，被任务数据淹没 → 需要单独配比或分阶段训练
- 流式输出的汉字乱码：`tokenizer.decode([token])` 逐 token 解码会把跨 token 的汉字打碎，已改成"解码整段、只打印新增部分"

---

## 九、9.12 续：人格 vs 任务 —— 亲手做出灾难性遗忘

给 `chat_sft.py` 加了 `--init-source sft --init-tag <tag>`（从已有对话模型续训，即分阶段 SFT），做了四组对照：

| 方案 | 分类准确率 | 闲聊"你好" |
|---|---|---|
| v7 单阶段（人格 4.7%） | 58.5% | ❌ "时尚。" |
| v7 → 纯人格续训 60 步 | **9.0%**（崩） | ✅ "你好！很高兴见到你，有什么可以帮你？" |
| v7 → 人格 + 分类 replay 各半，60 步 | **63.0%** | ❌ "推理时重复计算。" |
| v7 → 纯人格 + 低 LR，40 步 | 48.0% | ❌ 退化输出 |

**三个理解**：

1. **灾难性遗忘不是理论，是 60 步就能发生的事**：纯人格训练把人脸识别式的分类能力从 58.5% 打到 9.0%（=随机猜）。8.25 在 GPU 上看到的 bpb 3.27→12.91 就是这个现象
2. **replay（混合重放）保任务、不保新能力**：混入一半分类数据后任务能力反而更高（63.0%），但人格学不会——因为人格样本短，监督量被长文本任务稀释
3. **低 LR 只减缓、不解决**：48.0% 说明遗忘慢了，但人格依然没学会

**为什么真实模型没这个问题**：容量大（7B vs 40M）+ LoRA 只动小部分参数 + 系统提示/任务模板把不同能力在输入上区分开 + 数据量级完全不同。

**排查记录**：从 SFT checkpoint 续训时报 `AssertionError: total_batch_size (256) must be a multiple of 2048`——因为 checkpoint 的 `meta` 里存着 `max_seq_len=2048`，而模型实际是 256。已在脚本里加兜底：超过模型 `config.sequence_len` 就下调。


