# 推理性能基准（TTFT / TPOT）

- 模型：`sft/d8`，40.7M 参数，8 层，KV 头 4，dtype torch.float32
- 设备：cpu，decode 步数 32，取 3 次中位数
- KV Cache 占用：32.0 KB/token，seq=256 时 8.00 MB/序列

| prompt 长度 | TTFT (ms) | TPOT cache (ms/tok) | TPOT 全量重算 (ms/tok) | decode 加速 |
|---|---|---|---|---|
| 32 | 20.9 | 14.31 | 23.46 | **1.64x** |
| 64 | 30.1 | 14.24 | 30.46 | **2.14x** |
| 128 | 50.1 | 14.12 | 45.33 | **3.21x** |
| 192 | 53.7 | 13.71 | 57.20 | **4.17x** |
