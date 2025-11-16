#!/usr/bin/env python
"""
Train Qwen2.5-1.5B on a text-to-SQL dataset, using the same data and format
as a smaller fine-tuned Qwen2.5-0.5B model.

This script demonstrates "true" weight-level transfer learning:
- Start from a pre-trained large Qwen2.5-1.5B-Instruct model.
- Fine-tune it on a text-to-SQL dataset (prompt, sql).
- Optionally load a smaller fine-tuned teacher model just for comparison.

Usage example:

  python train_qwen25_transfer.py \
      --train_csv train_text2sql.csv \
      --output_dir ./models/qwen2_5_1_5b_text2sql \
      --teacher_dir ./models/qwen2_5_0_5b_text2sql

"""

import argparse
import os
import random

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Transfer learning on Qwen2.5-1.5B for text-to-SQL")

    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Path to CSV with at least columns: 'prompt' and 'sql'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/qwen2_5_1_5b_text2sql",
        help="Where to save the fine-tuned big Qwen model",
    )
    parser.add_argument(
        "--teacher_dir",
        type=str,
        default=None,
        help="Optional path to previously fine-tuned Qwen2.5-0.5B model (for comparison/demo only)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base big Qwen model to fine-tune (e.g. Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=1.0,
        help="Number of training epochs (1–2 recommended for big models)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=20,
        help="Number of random samples to use for quick exact-match evaluation",
    )

    return parser.parse_args()


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    # Expect columns: 'prompt', 'sql'
    if "prompt" not in df.columns or "sql" not in df.columns:
        raise ValueError("CSV must contain 'prompt' and 'sql' columns.")

    # Optional: filter languages or complexity here if needed
    # e.g., df = df[df["language"].isin(["en", "hinglish"])]

    def format_example(row):
        return f"User: {row['prompt']}\nSQL: {row['sql']}"

    df["text"] = df.apply(format_example, axis=1)

    dataset = Dataset.from_pandas(df[["prompt", "sql", "text"]])

    # 90/10 split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = dataset["train"]
    eval_ds = dataset["test"]

    return train_ds, eval_ds


def tokenize_dataset(train_ds, eval_ds, tokenizer, max_length):
    def tokenize_fn(batch):
        out = tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["prompt", "sql", "text"])
    eval_tok = eval_ds.map(tokenize_fn, batched=True, remove_columns=["prompt", "sql", "text"])

    return train_tok, eval_tok


def data_collator(features):
    # Simple collator for causal LM
    import torch

    batch = {}
    for k in features[0].keys():
        batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)
    return batch


def demo_teacher_vs_student(teacher_dir, train_ds):
    """Optional small demo: show teacher predictions for a few examples."""
    if teacher_dir is None:
        print("\n[Info] No teacher_dir provided, skipping teacher demo.\n")
        return

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"\n[Info] Loading teacher model from: {teacher_dir}")
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_dir, trust_remote_code=True)
        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        teacher_model.eval()
    except Exception as e:
        print(f"[Warning] Could not load teacher model: {e}")
        return

    def teacher_generate(text, max_new_tokens=64):
        inp = teacher_tokenizer(text, return_tensors="pt").to(teacher_model.device)
        with torch.no_grad():
            out_ids = teacher_model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=teacher_tokenizer.eos_token_id,
            )
        full = teacher_tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if "SQL:" in full:
            sql_part = full.split("SQL:", 1)[1].strip()
        else:
            sql_part = full.strip()
        return sql_part

    print("\n[Teacher Demo] Showing a few teacher predictions:\n")
    samples = random.sample(list(train_ds), k=min(3, len(train_ds)))
    for ex in samples:
        text = ex["text"]
        gold = ex["sql"]
        pred = teacher_generate(text)
        print("Prompt :", ex["prompt"])
        print("Gold SQL:", gold)
        print("Teacher SQL:", pred)
        print("-" * 80)


def evaluate_exact_match(model, tokenizer, eval_ds, num_samples=20, max_length=256):
    """Quick exact-match evaluation on a random subset of the eval set."""
    model.eval()
    samples = random.sample(list(eval_ds), k=min(num_samples, len(eval_ds)))

    def generate_sql(nl_prompt, max_new_tokens=64):
        input_text = f"User: {nl_prompt}\nSQL:"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        full = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if "SQL:" in full:
            sql_part = full.split("SQL:", 1)[1].strip()
        else:
            sql_part = full.strip()
        return sql_part, full

    correct = 0
    total = 0

    print("\n[Evaluation] Small exact-match evaluation on random subset:\n")
    for ex in samples:
        nl = ex["prompt"]
        gold_sql = str(ex["sql"]).strip()
        pred_sql, full = generate_sql(nl, max_new_tokens=64)
        pred_sql = pred_sql.strip()

        is_match = (pred_sql == gold_sql)
        correct += int(is_match)
        total += 1

        print("NL       :", nl)
        print("GOLD SQL :", gold_sql)
        print("PRED SQL :", pred_sql)
        print("MATCH    :", is_match)
        print("-" * 100)

    acc = correct / max(total, 1)
    print(f"\n[Evaluation] Exact-match accuracy on {total} samples: {acc:.2%}\n")
    return acc


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=== Qwen2.5 Transfer Learning Script ===")
    print(f"Train CSV : {args.train_csv}")
    print(f"Output dir: {args.output_dir}")
    if args.teacher_dir:
        print(f"Teacher   : {args.teacher_dir}")
    print(f"Base model: {args.model_name}")
    print("========================================\n")

    # 1) Load dataset
    print("[Step 1] Loading dataset...")
    train_ds, eval_ds = load_dataset(args.train_csv)
    print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

    # 2) Optional: show teacher predictions
    demo_teacher_vs_student(args.teacher_dir, train_ds)

    # 3) Load tokenizer & big Qwen model (student)
    print("\n[Step 2] Loading big Qwen model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("[Info] Model loaded.")

    # 4) Tokenize datasets
    print("\n[Step 3] Tokenizing dataset for student model...")
    train_tok, eval_tok = tokenize_dataset(train_ds, eval_ds, tokenizer, args.max_length)
    print("Tokenization done.")

    # 5) Training
    print("\n[Step 4] Starting fine-tuning on big Qwen model...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
    )

    trainer.train()
    print("\n[Info] Training finished.")

    # 6) Save final model
    print("\n[Step 5] Saving fine-tuned big Qwen model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[Info] Model saved to: {args.output_dir}")

    # 7) Quick evaluation on random subset
    _ = evaluate_exact_match(model, tokenizer, eval_ds, num_samples=args.eval_samples)

    print("\n=== Done. This big Qwen model is now fine-tuned for text-to-SQL. ===")


if __name__ == "__main__":
    main()
