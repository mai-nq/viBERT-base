# mai-roberta-base

A Vietnamese RoBERTa-based language model pre-trained on CC-100 Vietnamese and custom Vietnamese corpus.

## Model Description

- **Architecture**: BERT-base (RoBERTa-style pre-training)
- **Hidden size**: 768
- **Attention heads**: 12
- **Hidden layers**: 12
- **Vocab size**: 41,035
- **Max sequence length**: 512
- **Parameters**: ~110M

## Training Data

- CC-100 Vietnamese
- Custom Vietnamese corpus

## Usage

```python
from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mainguyen9/mai-roberta-base")
model = AutoModel.from_pretrained("mainguyen9/mai-roberta-base")

# Encode text
text = "Xin chào Việt Nam"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# Get embeddings
last_hidden_state = outputs.last_hidden_state
```

### For Masked Language Modeling

```python
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="mainguyen9/mai-roberta-base")
result = fill_mask("Hà Nội là [MASK] đô của Việt Nam.")
print(result)
```

## Benchmark Results

See [benchmark/](benchmark/) for evaluation scripts and detailed results.

| Task | Dataset | Metric | Score |
|------|---------|--------|-------|
| NER | PhoNER_COVID19 | F1 | **89.38** |
| NLI | XNLI Vietnamese | Accuracy | - |

### NER Performance Details (PhoNER_COVID19)

| Entity Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| AGE | 90.91 | 97.27 | 93.98 |
| DATE | 98.20 | 99.17 | 98.68 |
| GENDER | 89.96 | 92.23 | 91.08 |
| JOB | 66.59 | 51.75 | 58.24 |
| LOCATION | 88.52 | 91.33 | 89.90 |
| NAME | 94.09 | 90.56 | 92.29 |
| ORGANIZATION | 77.02 | 78.05 | 77.53 |
| PATIENT_ID | 95.61 | 98.54 | 97.05 |
| SYMPTOM_AND_DISEASE | 82.84 | 74.70 | 78.56 |
| TRANSPORTATION | 85.63 | 91.41 | 88.43 |

## Citation

```bibtex
@misc{mai-roberta-base,
  author = {Mai Nguyen},
  title = {mai-roberta-base: A Vietnamese RoBERTa Model},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/mainguyen9/mai-roberta-base}
}
```

## License

MIT License
