"""
Benchmark mai-roberta-base on Vietnamese NER task.
Dataset: PhoNER_COVID19 from GitHub
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def download_phoner_from_github():
    """Download PhoNER_COVID19 from GitHub and convert to HF dataset format."""
    import urllib.request

    base_url = "https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/word/"
    splits = {"train": "train_word.conll", "validation": "dev_word.conll", "test": "test_word.conll"}

    datasets = {}
    for split_name, filename in splits.items():
        url = base_url + filename
        print(f"  Downloading {split_name} from {url}...")

        with urllib.request.urlopen(url) as response:
            content = response.read().decode("utf-8")

        sentences = []
        tags = []
        current_words = []
        current_tags = []

        for line in content.strip().split("\n"):
            line = line.strip()
            if line == "":
                if current_words:
                    sentences.append(current_words)
                    tags.append(current_tags)
                    current_words = []
                    current_tags = []
            else:
                # Try both tab and space as delimiter
                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts) >= 2:
                    current_words.append(parts[0])
                    current_tags.append(parts[-1])

        if current_words:
            sentences.append(current_words)
            tags.append(current_tags)

        datasets[split_name] = Dataset.from_dict({
            "tokens": sentences,
            "ner_tags_str": tags
        })

    return DatasetDict(datasets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../mai-roberta-base")
    parser.add_argument("--output_dir", type=str, default="./ner_output")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset from GitHub
    print("Loading PhoNER_COVID19 dataset from GitHub...")
    dataset = download_phoner_from_github()

    # Get all unique tags
    all_tags = set()
    for split in dataset:
        for example in dataset[split]:
            all_tags.update(example["ner_tags_str"])

    # Create label mappings - standard BIO format
    label_list = sorted(list(all_tags))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    print(f"Labels ({num_labels}): {label_list}")
    print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} | Test: {len(dataset['test'])}")

    # Load tokenizer and model
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    def tokenize_and_align_labels(examples):
        """Tokenize inputs and align labels with tokens."""
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=256,
            padding="max_length",
        )

        all_labels = []
        for i, tags in enumerate(examples["ner_tags_str"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            tag_ids = [label2id[t] for t in tags]
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(tag_ids[word_idx])
                else:
                    label_ids.append(tag_ids[word_idx])
                previous_word_idx = word_idx
            all_labels.append(label_ids)

        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(p):
        """Compute NER metrics."""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

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
        metric_for_best_model="f1",
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
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(args.output_dir) / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Generate detailed classification report
    predictions = trainer.predict(tokenized_dataset["test"])
    preds = np.argmax(predictions.predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, predictions.label_ids)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, predictions.label_ids)
    ]

    report = classification_report(true_labels, true_predictions, digits=4)
    print("\nClassification Report:")
    print(report)

    with open(Path(args.output_dir) / "classification_report.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
