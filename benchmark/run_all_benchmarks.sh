#!/bin/bash
# Run all benchmarks for mai-roberta-base

set -e

MODEL_PATH="../mai-roberta-base"
RESULTS_DIR="./results"

mkdir -p $RESULTS_DIR

echo "=========================================="
echo "Running mai-roberta-base Benchmarks"
echo "=========================================="

# NER - PhoNER_COVID19
echo ""
echo "[1/4] Running NER benchmark (PhoNER_COVID19)..."
python ner_phoner.py \
    --model_path $MODEL_PATH \
    --output_dir $RESULTS_DIR/ner \
    --batch_size 16 \
    --epochs 5

# Sentiment - UIT-VSFC
echo ""
echo "[2/4] Running Sentiment benchmark (UIT-VSFC)..."
python sentiment_vsfc.py \
    --model_path $MODEL_PATH \
    --output_dir $RESULTS_DIR/sentiment \
    --batch_size 16 \
    --epochs 5

# Emotion Classification - UIT-VSMEC
echo ""
echo "[3/4] Running Emotion Classification benchmark (UIT-VSMEC)..."
python text_classification_vsmec.py \
    --model_path $MODEL_PATH \
    --output_dir $RESULTS_DIR/vsmec \
    --batch_size 16 \
    --epochs 5

# NLI - XNLI Vietnamese
echo ""
echo "[4/4] Running NLI benchmark (XNLI Vietnamese)..."
python nli_vinli.py \
    --model_path $MODEL_PATH \
    --output_dir $RESULTS_DIR/nli \
    --batch_size 16 \
    --epochs 5

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "Results saved in $RESULTS_DIR"
echo "=========================================="

# Collect and display results
echo ""
echo "Summary of Results:"
echo "==================="

for task in ner sentiment vsmec nli; do
    if [ -f "$RESULTS_DIR/$task/test_results.json" ]; then
        echo ""
        echo "[$task]"
        cat "$RESULTS_DIR/$task/test_results.json"
    fi
done
