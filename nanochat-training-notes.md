# nanochat 本地训练笔记（2026-08-23）

> 目标：在本地（CPU + 4060）跑通 nanochat 训练全流程，作为 9 月项目（从零 LLM 推理系统）的底座。
> 成果：✅ 训练成功（loss 下降 + 模型保存），本地新闻数据喂模型。

---

## 一、环境（本地 Windows + conda ai 环境）

- Python 3.11 + torch 2.5.1（CPU）+ transformers 5.15.1
- nanochat clone 到 `C:\Users\耶\workspace\nanochat`
- 用 conda ai 环境跑（绕开 uv 慢下载），缺啥装啥：
  ```
  pip install rustbpe tiktoken pyarrow numpy tqdm wandb
  ```

## 二、改造（我的增量：支持本地文本训练）

### 1. 分词器：tok_train.py 改读本地文本
- `scripts/tok_train.py` 的 `text_iterator()` 改为读本地 `test.txt`
- 绕开 400B parquet 数据集下载（climbmix 太大 + HF 证书问题）


### 2. dataloader：加本地文本双模式
- `nanochat/dataloader.py` 新增：
  - `list_text_files(data_dir)`：找目录/文件里的 .txt
  - `txt_to_docs(text, doc_max_char, line_mode)`：按空行或按行切文档，自动去标签（cnews 格式 `标签\t正文`）
  - `_text_document_batches()`：本地文本版数据迭代（复刻 parquet 版的 resume/epoch 结构）
- `tokenizing_distributed_data_loader_with_state_bos_bestfit()` 加 `text_path=None` 参数：设了走本地，不设走 parquet

### 3. base_train.py：加 --text-path 参数
- 参数区加 `--text-path`
- 两处 dataloader 调用（train/val）传 `text_path=args.text_path`

## 三、卡点排查记录（重要！debug 经验）

### 卡点 1：torch.compile 失败（cl not found）
- **症状**：`RuntimeError: Compiler: cl is not found`（Windows 无 MSVC cl.exe）
- **解决**：注释掉所有 `torch.compile`：
  - `scripts/base_train.py` 的 `model = torch.compile(...)`
  - `nanochat/optim.py` 的 2 处 `@torch.compile` 装饰器（第 23、111 行）

### 卡点 2：训练"卡死"（CPU 22% 无输出）
- **排查过程**（分段测试排除）：
  1. tokenizer 加载 ✅ 快
  2. dataloader 第一笔 ✅ 快
  3. 模型前向 ✅ 0.0 秒
  4. 模型反向 ✅ 0.03 秒
  5. setup_optimizer ✅ 1.31 秒
  6. **真相**：卡在训练结束的**评估环节**（evaluate_core 下载 eval_bundle / sample 跑 engine）
- **解决**：训练时关掉评估/采样/保存：
  ```
  --eval-every -1 --core-metric-every -1 --sample-every -1 --save-every -1
  ```
- **经验**：CPU 上"卡死"先怀疑"网络/下载/生成"环节（eval_bundle 下载、engine 生成），不是训练本身

### 卡点 3：faulthandler 定位工具
- `faulthandler.dump_traceback_later(30, exit=True)`：超时打印卡住位置
- 这次抓到 `apply_rotary_emb`——但那是"正在执行的某步"，不是真卡点（误导过一次）

## 四、训练成功命令

```powershell
cd C:\Users\耶\workspace\nanochat
C:\Users\Public\miniconda3\envs\ai\python.exe -m scripts.base_train --depth=2 --max-seq-len=64 --device-batch-size=1 --total-batch-size=64 --num-iterations=5 --run=dummy --text-path=test.txt --window-pattern L --eval-every -1 --core-metric-every -1 --sample-every -1 --save-every -1
```

**结果**：
```
step 00000/00005 | loss: 5.943285
step 00004/00005 | loss: 5.940122   ← loss 下降
Saved model to base_checkpoints/d2/
Total training time: 0.00m
```

## 五、数据准备

- cnews 新闻数据（中文，10 类）：`cnews.train.txt`（130MB 完整）+ `cnews_small.txt`（前 5000 行，12.7MB）
- 格式：`标签\t正文`（每行一篇）→ `txt_to_docs(line_mode=True)` 处理
- test.txt（15KB 自造）用于快速验证

## 六、下一步（待做）

1. 训练久一点（100+ 步 + cnews_small）验证 loss 持续下降
2. 处理评估环节（eval_bundle 下载问题）——跑 base_eval
3. 对话（chat_cli）——应用 demo
4. 把这些改动整理成"本地文本训练扩展"文档（简历增量）

## 七、简历增量（这段改动怎么描述）

> **nanochat 扩展：支持本地文本训练**——为 nanochat 增加本地文本数据加载（dataloader 双模式：parquet/本地 txt）、自定义分词器训练入口、--text-path 训练参数；绕开 400B 云端数据集，可在单机 CPU 上跑通完整训练流程。
