---
language:
- vi
license: mit
tags:
- bert
- roberta
- vietnamese
- fill-mask
- feature-extraction
datasets:
- cc100
pipeline_tag: fill-mask
model-index:
- name: viBERT-base
  results:
  - task:
      type: token-classification
      name: Named Entity Recognition
    dataset:
      name: PhoNER_COVID19
      type: VinAIResearch/PhoNER_COVID19
    metrics:
    - type: f1
      value: 89.38
      name: F1 (micro)
  - task:
      type: natural-language-inference
      name: Natural Language Inference
    dataset:
      name: XNLI Vietnamese
      type: xnli
    metrics:
    - type: accuracy
      value: 71.06
      name: Accuracy
  - task:
      type: text-classification
      name: Hate Speech Detection
    dataset:
      name: ViHSD
      type: visolex/ViHSD
    metrics:
    - type: accuracy
      value: 87.89
      name: Accuracy
    - type: f1
      value: 65.63
      name: F1 (macro)
---

# viBERT-base

A Vietnamese RoBERTa-based language model pre-trained on CC-100 Vietnamese and custom Vietnamese corpus.

## Model Description

**viBERT-base** is a BERT-base architecture model trained with RoBERTa-style pre-training on Vietnamese text data. It can be used for various Vietnamese NLP downstream tasks such as Named Entity Recognition, Text Classification, Question Answering, and more.

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | BERT-base |
| Hidden size | 768 |
| Attention heads | 12 |
| Hidden layers | 12 |
| Vocab size | 41,035 |
| Max sequence length | 512 |
| Parameters | ~110M |

## Training Data

- **CC-100 Vietnamese**: Large-scale web crawl data
- **Custom Vietnamese corpus**: Additional curated Vietnamese text

## Usage

### Feature Extraction

```python
from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mainguyen9/viBERT-base")
model = AutoModel.from_pretrained("mainguyen9/viBERT-base")

# Encode text
text = "Xin chào Việt Nam"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# Get embeddings
last_hidden_state = outputs.last_hidden_state
```

### Masked Language Modeling

```python
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="mainguyen9/viBERT-base")
result = fill_mask("Hà Nội là [MASK] đô của Việt Nam.")
print(result)
```

### Fine-tuning for NER

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained(
    "mainguyen9/viBERT-base",
    num_labels=num_labels
)
tokenizer = AutoTokenizer.from_pretrained("mainguyen9/viBERT-base")
```

## Benchmark Results

| Task | Dataset | Metric | Score |
|------|---------|--------|-------|
| NER | PhoNER_COVID19 | F1 | **89.38** |
| NLI | XNLI Vietnamese | Accuracy | **71.06** |
| Hate Speech | ViHSD | Accuracy | **87.89** |

### NER Performance Details (PhoNER_COVID19)

Fine-tuned with 5 epochs, batch size 32, learning rate 2e-5.

| Entity Type | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| AGE | 90.91 | 97.27 | 93.98 | 586 |
| DATE | 98.20 | 99.17 | 98.68 | 3,026 |
| GENDER | 89.96 | 92.23 | 91.08 | 476 |
| JOB | 66.59 | 51.75 | 58.24 | 570 |
| LOCATION | 88.52 | 91.33 | 89.90 | 10,845 |
| NAME | 94.09 | 90.56 | 92.29 | 1,388 |
| ORGANIZATION | 77.02 | 78.05 | 77.53 | 1,640 |
| PATIENT_ID | 95.61 | 98.54 | 97.05 | 2,120 |
| SYMPTOM_AND_DISEASE | 82.84 | 74.70 | 78.56 | 2,158 |
| TRANSPORTATION | 85.63 | 91.41 | 88.43 | 489 |
| **Micro Average** | **89.09** | **89.69** | **89.38** | **23,298** |

### NLI Performance (XNLI Vietnamese)

Fine-tuned with 5 epochs, batch size 64, learning rate 2e-5.

| Metric | Score |
|--------|-------|
| Accuracy | **71.06%** |
| F1 (macro) | **71.02%** |

### Hate Speech Detection (ViHSD)

Fine-tuned with 5 epochs, batch size 8, learning rate 2e-5.

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| CLEAN | 91.84% | 96.40% | 94.06% | 5,548 |
| OFFENSIVE | 51.86% | 40.77% | 45.65% | 444 |
| HATE | 67.32% | 49.71% | 57.19% | 688 |
| **Accuracy** | | | **87.89%** | 6,680 |
| **Macro Avg** | **70.34%** | **62.29%** | **65.63%** | 6,680 |

## Limitations

- Primarily trained on Vietnamese text; performance may vary for code-mixed text
- 512 token maximum sequence length

## Citation

```bibtex
@misc{vibert-base,
  author = {Mai Nguyen},
  title = {viBERT-base: A Vietnamese RoBERTa Model},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/mainguyen9/viBERT-base}
}
```

## License

MIT License
