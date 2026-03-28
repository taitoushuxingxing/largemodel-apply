"""
测试集评估脚本
在测试集上逐条推理，计算准确率、F1、分类报告、混淆矩阵
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from tqdm import tqdm

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "outputs/news_cls_v1/best_model"
TEST_FILE  = "data/processed/test.json"


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, sample):
    """单条推理"""
    messages = [
        {"role": "system", "content": sample["instruction"]},
        {"role": "user",   "content": sample["input"]},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=20, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    print("=" * 60)
    print("📊 新闻分类 - 模型评估")
    print("=" * 60)

    # 加载
    print("\n📥 加载模型...")
    model, tokenizer = load_model(MODEL_PATH)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"📋 测试集: {len(test_data)} 条")

    # 逐条预测
    true_labels, pred_labels, errors = [], [], []
    for sample in tqdm(test_data, desc="🔄 评估中"):
        true = sample["output"]
        pred = predict(model, tokenizer, sample)
        true_labels.append(true)
        pred_labels.append(pred)
        if pred != true:
            errors.append({"input": sample["input"][:80], "true": true, "pred": pred})

    # ============================================================
    # 评估指标
    # ============================================================
    acc         = accuracy_score(true_labels, pred_labels)
    macro_f1    = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    macro_p     = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    macro_r     = recall_score(true_labels, pred_labels, average="macro", zero_division=0)

    print(f"\n{'=' * 60}")
    print("📈 总体指标")
    print(f"{'=' * 60}")
    print(f"  准确率 (Accuracy):      {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  宏平均 F1 (Macro-F1):   {macro_f1:.4f}")
    print(f"  加权 F1 (Weighted-F1):  {weighted_f1:.4f}")
    print(f"  宏平均精确率:            {macro_p:.4f}")
    print(f"  宏平均召回率:            {macro_r:.4f}")

    # 分类别详细报告
    print(f"\n{'=' * 60}")
    print("📋 分类别详细报告")
    print(f"{'=' * 60}")
    print(classification_report(true_labels, pred_labels, zero_division=0))

    # 混淆矩阵
    labels = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    print(f"{'=' * 60}")
    print("🔢 混淆矩阵")
    print(f"{'=' * 60}")
    print(f"标签顺序: {labels}")
    print(cm)

    # 错误案例
    print(f"\n{'=' * 60}")
    print(f"❌ 错误案例 ({len(errors)} / {len(test_data)} 条)")
    print(f"{'=' * 60}")
    for i, case in enumerate(errors[:20]):  # 显示前20个
        print(f"  [{i+1:>2}] 输入: {case['input']}")
        print(f"       真实: {case['true']}  |  预测: {case['pred']}")
        print()

    # 保存结果
    eval_result = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "total": len(test_data),
        "errors": len(errors),
        "error_cases": errors,
    }
    result_path = os.path.join(MODEL_PATH, "eval_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    print(f"💾 评估结果已保存: {result_path}")
    print("\n✅ 评估完成！")


if __name__ == "__main__":
    main()