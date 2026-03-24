"""
Benchmark mai-roberta-base on UIT-VSFC Sentiment Classification task.
Dataset: uitnlp/vietnamese_students_feedback
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def tokenize_function(examples, tokenizer, max_length=256):
    """Tokenize text inputs."""
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def compute_metrics(eval_pred):
    """Compute classification metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="macro"),
        "recall": recall_score(labels, predictions, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../mai-roberta-base")
    parser.add_argument("--output_dir", type=str, default="./sentiment_output")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    print("Loading UIT-VSFC dataset...")
    dataset = load_dataset("uitnlp/vietnamese_students_feedback")

    # Get label info - sentiment labels: 0=negative, 1=neutral, 2=positive
    label_list = ["negative", "neutral", "positive"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    print(f"Labels: {label_list}")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
    print(f"Test size: {len(dataset['test'])}")

    # Load tokenizer and model
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["sentence"],
    )
    tokenized_dataset = tokenized_dataset.rename_column("sentiment", "labels")

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Training...")
    trainer.train()

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"Test Results: {test_results}")

    # Save results
    results_path = Path(args.output_dir) / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
