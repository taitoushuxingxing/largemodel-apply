"""
数据处理脚本：清洗 + 构建 prompt + 划分训练/验证/测试集
输入: data/raw/news_intent_data.csv
输出: data/processed/train.json, val.json, test.json, label_map.json
"""

import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from collections import Counter

# ============================================================
# 1. 加载原始数据
# ============================================================
print("=" * 55)
print("📦 数据处理与划分")
print("=" * 55)

df = pd.read_csv("data/raw/news_intent_data.csv")
print(f"\n📥 加载原始数据: {len(df)} 条")
print(f"   列名: {list(df.columns)}")
print(f"\n   标签分布:")
for intent, cnt in df["intent"].value_counts().items():
    print(f"     {intent:>15}: {cnt} 条")

# ============================================================
# 2. 数据清洗
# ============================================================
print("\n🧹 数据清洗...")

# 去空值
before = len(df)
df = df.dropna(subset=["query", "intent"])
print(f"   去空值: {before} → {len(df)}")

# 去首尾空格
df["query"] = df["query"].str.strip()
df["intent"] = df["intent"].str.strip()

# 去重
before = len(df)
df = df.drop_duplicates(subset=["query"])
print(f"   去重:   {before} → {len(df)}")

# ============================================================
# 3. 构建模型输入格式 (instruction + input + output)
# ============================================================
print("\n📝 构建 prompt 格式...")

SYSTEM_PROMPT = (
    "你是一个新闻分类助手。根据给定的新闻标题，判断它属于哪个类别。"
    "只返回类别标签，不要返回其他内容。"
    "可选类别：sports(体育)、technology(科技)、entertainment(娱乐)、finance(财经)、military(军事)"
)

def build_sample(row):
    return {
        "instruction": SYSTEM_PROMPT,
        "input": f"请判断以下新闻的类别：\n{row['query']}",
        "output": row["intent"],
    }

samples = df.apply(build_sample, axis=1).tolist()
print(f"   总样本数: {len(samples)}")

# 打印一条样例
print(f"\n   📋 样例:")
print(f"   instruction: {samples[0]['instruction'][:60]}...")
print(f"   input:       {samples[0]['input']}")
print(f"   output:      {samples[0]['output']}")

# ============================================================
# 4. 数据集划分 (8:1:1, 分层抽样)
# ============================================================
print("\n✂️  数据集划分 (训练:验证:测试 = 8:1:1)...")

labels = [s["output"] for s in samples]

# 先分出测试集 10%
train_val_samples, test_samples, train_val_labels, _ = train_test_split(
    samples, labels, test_size=0.1, random_state=42, stratify=labels
)

# 再从剩余中分出验证集 (占原始 10%, 即剩余的 1/9)
train_samples, val_samples = train_test_split(
    train_val_samples, test_size=1 / 9, random_state=42, stratify=train_val_labels
)

print(f"   训练集: {len(train_samples)} 条 ({len(train_samples)/len(samples)*100:.1f}%)")
print(f"   验证集: {len(val_samples)} 条 ({len(val_samples)/len(samples)*100:.1f}%)")
print(f"   测试集: {len(test_samples)} 条 ({len(test_samples)/len(samples)*100:.1f}%)")

# 验证各集合中标签分布
print(f"\n   训练集标签分布:")
for intent, cnt in sorted(Counter(s["output"] for s in train_samples).items()):
    print(f"     {intent:>15}: {cnt}")

# ============================================================
# 5. 保存
# ============================================================
print("\n💾 保存处理后的数据...")

output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

save_json(train_samples, f"{output_dir}/train.json")
save_json(val_samples, f"{output_dir}/val.json")
save_json(test_samples, f"{output_dir}/test.json")

# 标签映射
intent_labels = sorted(df["intent"].unique().tolist())
label2id = {label: idx for idx, label in enumerate(intent_labels)}
id2label = {str(idx): label for label, idx in label2id.items()}
save_json({"label2id": label2id, "id2label": id2label, "num_labels": len(intent_labels)},
          f"{output_dir}/label_map.json")

print(f"   ✅ {output_dir}/train.json")
print(f"   ✅ {output_dir}/val.json")
print(f"   ✅ {output_dir}/test.json")
print(f"   ✅ {output_dir}/label_map.json")

print(f"\n   标签映射: {label2id}")
print("\n✅ 数据准备完成！接下来运行 scripts/train.py 开始训练")