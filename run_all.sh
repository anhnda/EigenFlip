#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
OUTPUT_DIR=./quantized_models/eigenflip_3bit
LOG_DIR=./logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eigenflip_3bit_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== run started $(date) ==="
echo "log: $LOG_FILE"
#none clc eigenflipeigenflip_solve
for ENC in gptq; do
  CELL_DIR="$OUTPUT_DIR/rtn_${ENC}"

  # layer-batch-size per encoder: Gram-heavy ones need smaller batches / cpu eigh
  case "$ENC" in
    none|clc)                   LBS=16; EXTRA="" ;;
    eigenflip|eigenflip_solve)  LBS=8;  EXTRA="" ;;
    gptq)                       LBS=4;  EXTRA="--eig-on-cpu" ;;
  esac

  echo
  echo "############################################################"
  echo "# encoder=$ENC  lbs=$LBS  ($(date))"
  echo "############################################################"

  echo ">>> [1/3] quantizing rtn+$ENC"
  PYTHONPATH=. python eigenflip/run_fast.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --bits 3 --group-size 128 --k 16 \
    --base rtn --encoder "$ENC" \
    --calib-dataset c4 --n-calib 128 --seqlen 2048 \
    --layer-batch-size $LBS $EXTRA

  if [ ! -d "$CELL_DIR" ]; then
    echo "!!! expected checkpoint missing: $CELL_DIR -- skipping eval/delete"
    continue
  fi

  echo ">>> [2/3] eval_ppl on $CELL_DIR"
  PYTHONPATH=. python eval_ppl.py \
    --model-path "$CELL_DIR" \
    --datasets wikitext2 c4 --seqlen 2048

  echo ">>> [2.5] preserving ppl.json"
  cp "$CELL_DIR/ppl.json" "$OUTPUT_DIR/rtn_${ENC}_ppl.json" 2>/dev/null || true

  echo ">>> [3/3] deleting $CELL_DIR"
  rm -rf "$CELL_DIR"
  echo "<<< done rtn+$ENC"
done

echo
echo "=== all cells done $(date) ==="
echo "preserved ppl files: $OUTPUT_DIR/rtn_*_ppl.json"
