"""
local_eval.py —— 本地训练器评估脚本
评估已训练模型：bpb（每字节比特数）+ 对话质量观察
用法：python local_eval.py --model-tag d2
"""
import argparse
import torch

from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer

def evaluate_bpb(model, tokenizer, text_path, max_tokens=16384):
    with open(text_path, encoding='utf-8') as f:
        text = f.read()
    # tokenize
    tokens = tokenizer.encode([text[:max_tokens]], prepend=tokenizer.get_bos_token_id())[0]
    tokens = tokens[:1280]
    # 模型前向算 loss
    x = torch.tensor([tokens[:-1]], device='cpu')
    y = torch.tensor([tokens[1:]], device='cpu')
    with torch.no_grad():
        loss = model(x, targets=y)   # 模型返回 loss
    bpb = loss.item() / 0.6931   # loss / ln(2)
    print(f'bpb: {bpb:.4f}')

def test_dialogue(model, tokenizer, questions):
    """对话测试：对问题生成回复"""
    model.eval()
    for q in questions:
        ids, mask = tokenizer.render_conversation({'messages': [{'role': 'user', 'content': q}]})
        with torch.no_grad():
            cur = list(ids)
            for _ in range(40):
                logits = model(torch.tensor([cur], device='cpu'))
                nxt = logits[0, -1].argmax().item()
                cur.append(nxt)
                if nxt == tokenizer.encode_special('<|assistant_end|>'):
                    break
        print(f'Q: {q}\nA: {tokenizer.decode(cur)}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-tag', type=str, default='d2')
    parser.add_argument('--text-path', type=str, default=None, help='bpb 评估用的文本')
    parser.add_argument('--split-tokens', type=int, default=16384)
    args = parser.parse_args()

    model, tokenizer, meta = load_model('sft', torch.device('cpu'), phase='eval', model_tag=args.model_tag)
    model.eval()

    print(f'===== 评估模型: {args.model_tag} =====')

    # ① bpb 评估
    if args.text_path:
        print('--- bpb 评估 ---')
        evaluate_bpb(model, tokenizer, args.text_path, args.split_tokens)

    # ② 对话测试
    print('--- 对话测试 ---')
    test_dialogue(model, tokenizer, ['你好', '什么是机器学习', '1加1等于几', '什么是Transformer'])

    print('===== 评估完成 =====')

if __name__ == '__main__':
    main()