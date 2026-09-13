"""KV Cache 正确性冒烟测试（CI 用，纯 CPU，不需要数据集和分词器）。

判定标准：同一段 token，
  1) 一次性全量前向的最后一位 logits
  2) 先 prefill 写入 cache、再喂 1 个 token 的最后一位 logits
两者必须一致（容差 1e-4）。这条测试同时覆盖 RoPE 绝对位置切片是否正确。
"""
import torch

from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig


def _tiny_model():
    cfg = GPTConfig(sequence_len=64, vocab_size=256, n_layer=2, n_head=2,
                    n_kv_head=2, n_embd=64, window_pattern="L")
    torch.manual_seed(0)
    model = GPT(cfg)
    model.eval()
    return model, cfg


def _kv_kwargs(cfg):
    return {"num_heads": cfg.n_kv_head, "head_dim": cfg.n_embd // cfg.n_head,
            "num_layers": cfg.n_layer}


@torch.no_grad()
def test_prefill_matches_full_forward():
    model, cfg = _tiny_model()
    tokens = torch.randint(0, cfg.vocab_size, (1, 16))

    full = model(tokens)
    cache = KVCache(batch_size=1, seq_len=32, device="cpu", dtype=torch.float32, **_kv_kwargs(cfg))
    pre = model(tokens, kv_cache=cache)

    assert torch.allclose(full[:, -1], pre[:, -1], atol=1e-4), \
        "prefill 的最后一位 logits 应与全量前向一致"


@torch.no_grad()
def test_decode_one_token_matches_full_forward():
    model, cfg = _tiny_model()
    tokens = torch.randint(0, cfg.vocab_size, (1, 16))
    nxt = torch.randint(0, cfg.vocab_size, (1, 1))

    cache = KVCache(batch_size=1, seq_len=32, device="cpu", dtype=torch.float32, **_kv_kwargs(cfg))
    model(tokens, kv_cache=cache)                    # prefill
    step = model(nxt, kv_cache=cache)                # decode：只喂 1 个 token

    full = model(torch.cat([tokens, nxt], dim=1))    # 全量重算
    assert torch.allclose(step[:, -1], full[:, -1], atol=1e-4), \
        "decode 1 个 token 的 logits 应与全量重算一致（位置编码切片错误会在这里暴露）"


@torch.no_grad()
def test_cache_keeps_growing():
    model, cfg = _tiny_model()
    cache = KVCache(batch_size=1, seq_len=32, device="cpu", dtype=torch.float32, **_kv_kwargs(cfg))
    tokens = torch.randint(0, cfg.vocab_size, (1, 4))
    model(tokens, kv_cache=cache)
    assert cache.get_pos() == 4, f"prefill 后 cache 位置应为 4，实际 {cache.get_pos()}"

    model(torch.randint(0, cfg.vocab_size, (1, 1)), kv_cache=cache)
    assert cache.get_pos() == 5, "每 decode 一个 token，cache 位置应 +1"
