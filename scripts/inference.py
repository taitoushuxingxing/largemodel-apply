"""
命令行单条推理脚本
接收用户从命令行输入的新闻文本，打印分类结果。
使用与 evaluate.py 相同的模型加载和推理逻辑。
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "outputs/news_cls_v1/best_model"

SYSTEM_PROMPT = (
    "你是一个新闻分类助手。根据给定的新闻标题，判断它属于哪个类别。"
    "只返回类别标签，不要返回其他内容。"
    "可选类别：sports(体育)、technology(科技)、entertainment(娱乐)、finance(财经)、military(军事)"
)


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, query: str) -> str:
    """对单条新闻文本执行推理，返回分类标签。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"请判断以下新闻的类别：\n{query}"},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="新闻分类命令行推理")
    parser.add_argument(
        "--model_path",
        default=MODEL_PATH,
        help=f"模型路径（默认: {MODEL_PATH}）",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="待分类的新闻文本（不提供则进入交互模式）",
    )
    args = parser.parse_args()

    print("=" * 55)
    print("📰 新闻分类 - 命令行推理")
    print("=" * 55)

    print(f"\n📥 加载模型: {args.model_path}")
    model, tokenizer = load_model(args.model_path)
    device = next(model.parameters()).device
    print(f"   设备: {device}\n")

    if args.query:
        result = predict(model, tokenizer, args.query)
        print(f"输入: {args.query}")
        print(f"分类: {result}")
    else:
        print("进入交互模式（输入 'q' 或 'quit' 退出）\n")
        while True:
            try:
                query = input("请输入新闻文本: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出")
                break
            if not query:
                continue
            if query.lower() in ("q", "quit"):
                break
            result = predict(model, tokenizer, query)
            print(f"分类结果: {result}\n")


if __name__ == "__main__":
    main()
