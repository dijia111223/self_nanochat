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

## 五、下一步（待做）

1. **更多对话数据**（几百~几千条不同问答）→ 模型从"背答案"到"泛化"
2. **更大模型**（depth 12+）+ 更多预训练数据 → bpb 更低
3. **本地 4060**：装 CUDA torch（网络问题待解决）→ 本地 GPU 训练省线
4. **chat_cli 对话接口**（SFT 模型 + 交互对话）
