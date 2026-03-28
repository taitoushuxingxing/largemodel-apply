"""
全参数微调训练脚本
模型: Qwen2.5-0.5B-Instruct
任务: 新闻五分类 (sports / technology / entertainment / finance / military)
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from torch.utils.data import Dataset

# ============================================================
# 1. 配置区（在这里修改所有参数）
# ============================================================

# --- 路径 ---
MODEL_PATH = "model/Qwen/Qwen2.5-0.5B-Instruct"       # 基座模型路径
TRAIN_FILE = "data/processed/train.json"
VAL_FILE   = "data/processed/val.json"
OUTPUT_DIR = "outputs/news_cls_v2"

# --- 超参数 ---
MAX_LENGTH = 256                    # 新闻标题不长，256 足���
NUM_EPOCHS = 10                     # 训练轮数
BATCH_SIZE = 8                      # 每GPU batch
GRAD_ACCUM = 4                      # 梯度累积 → 等效 batch = 8×4=32
LEARNING_RATE = 2e-5                # 全参微调推荐 1e-5 ~ 5e-5
WARMUP_RATIO = 0.1                  # 前 10% 步数做学习率预热
SEED = 42


# ============================================================
# 2. 数据集类
# ============================================================

class NewsDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 构建 chat 格式
        messages = [
            {"role": "system", "content": sample["instruction"]},
            {"role": "user",   "content": sample["input"]},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = input_text + sample["output"] + self.tokenizer.eos_token

        # Tokenize
        input_ids = self.tokenizer.encode(
            full_text, max_length=self.max_length, truncation=True
        )
        # 输入部分的 token 数（用于构建 labels mask）
        input_part_ids = self.tokenizer.encode(
            input_text, max_length=self.max_length, truncation=True
        )

        # labels: 输入部分填 -100（不算 loss），只在输出部分计算 loss
        labels = [-100] * len(input_part_ids) + input_ids[len(input_part_ids):]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


# ============================================================
# 3. 主训练流程
# ============================================================

def main():
    print("=" * 55)
    print("🚀 新闻分类 - 全参数微调训练")
    print("=" * 55)

    # --- 加载 Tokenizer ---
    print("\n📥 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 加载模型 ---
    print("📥 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.enable_input_require_grads()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数:   {total_params / 1e6:.1f}M")
    print(f"   可训练:   {trainable_params / 1e6:.1f}M (100%)")

    # --- 加载数据集 ---
    print("\n📥 加载数据集...")
    train_dataset = NewsDataset(TRAIN_FILE, tokenizer, MAX_LENGTH)
    val_dataset   = NewsDataset(VAL_FILE,   tokenizer, MAX_LENGTH)
    print(f"   训练集: {len(train_dataset)} 条")
    print(f"   验证集: {len(val_dataset)} 条")

    # --- 训练参数 ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        seed=SEED,
        report_to="tensorboard",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # --- 数据整理器 ---
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, max_length=MAX_LENGTH
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # --- 开始训练 ---
    print(f"\n{'=' * 55}")
    print(f"⚡ 开始训练")
    print(f"   等效 Batch Size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"   学习率: {LEARNING_RATE}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"{'=' * 55}\n")

    trainer.train()

    # --- 保存最佳模型 ---
    best_path = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    print(f"\n✅ 最佳模型已保存: {best_path}")
    print("✅ 训练完成！接下来运行 scripts/evaluate.py 评估效果")


if __name__ == "__main__":
    main()