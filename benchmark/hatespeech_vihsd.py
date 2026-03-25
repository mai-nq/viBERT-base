#!/usr/bin/env python3
"""
Hate Speech Detection benchmark on ViHSD dataset for Vietnamese language models.
Dataset: visolex/ViHSD - 33,400 comments labeled as CLEAN (0), OFFENSIVE (1), HATE (2)
"""

import argparse
import json
import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
    }


def main():
    parser = argparse.ArgumentParser(description='Hate Speech Detection benchmark on ViHSD')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--output_dir', type=str, default='./results/hatespeech', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    args = parser.parse_args()

    # Load dataset
    print("Loading ViHSD dataset...")
    dataset = load_dataset('visolex/ViHSD')

    # Split by type field
    train_ds = dataset['train'].filter(lambda x: x['type'] == 'train')
    dev_ds = dataset['train'].filter(lambda x: x['type'] == 'dev')
    test_ds = dataset['train'].filter(lambda x: x['type'] == 'test')

    # Labels: CLEAN (0), OFFENSIVE (1), HATE (2)
    label_names = ['CLEAN', 'OFFENSIVE', 'HATE']
    num_labels = len(label_names)

    print(f"Labels: {label_names}")
    print(f"Train size: {len(train_ds)}")
    print(f"Dev size: {len(dev_ds)}")
    print(f"Test size: {len(test_ds)}")

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=num_labels,
    )

    # Tokenize
    def tokenize_function(examples):
        # Handle None values
        texts = [str(t) if t is not None else "" for t in examples['free_text']]
        return tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    print("Tokenizing dataset...")
    train_tokenized = train_ds.map(tokenize_function, batched=True, remove_columns=['dataset', 'type', 'free_text'])
    dev_tokenized = dev_ds.map(tokenize_function, batched=True, remove_columns=['dataset', 'type', 'free_text'])
    test_tokenized = test_ds.map(tokenize_function, batched=True, remove_columns=['dataset', 'type', 'free_text'])

    # Rename label column
    train_tokenized = train_tokenized.rename_column('label_id', 'labels')
    dev_tokenized = dev_tokenized.rename_column('label_id', 'labels')
    test_tokenized = test_tokenized.rename_column('label_id', 'labels')

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        fp16=True,
        report_to='none',
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Training...")
    trainer.train()

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_tokenized)

    # Get predictions for detailed report
    predictions = trainer.predict(test_tokenized)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    # Classification report
    report = classification_report(labels, preds, target_names=label_names, digits=4)
    print("\nClassification Report:")
    print(report)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        'accuracy': float(test_results['eval_accuracy']),
        'f1_macro': float(test_results['eval_f1_macro']),
        'f1_weighted': float(test_results['eval_f1_weighted']),
        'precision': float(test_results['eval_precision']),
        'recall': float(test_results['eval_recall']),
    }

    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    print(f"\nTest Results: {test_results}")
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
