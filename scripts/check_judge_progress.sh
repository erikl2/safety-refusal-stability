#!/bin/bash
# Check progress of judge running on Lambda

source .env

echo "=================================="
echo "Judge Progress on Lambda H100"
echo "=================================="
echo ""

# Check if process is running
echo "[1] Process Status:"
ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "ps aux | grep 'python.*judge' | grep -v grep | wc -l | xargs -I {} echo '  Python judge processes running: {}'"

# Count completed label files
echo ""
echo "[2] Files Completed:"
LABEL_COUNT=$(ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "ls ~/safety-refusal-stability/data/results/labels/*_labels.csv 2>/dev/null | wc -l" | tr -d ' ')
TOTAL_FILES=20
echo "  Labeled: $LABEL_COUNT / $TOTAL_FILES files"

# Show recent log tail
echo ""
echo "[3] Recent Log Output:"
ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "tail -15 ~/safety-refusal-stability/logs/judge_all.log 2>/dev/null | grep -E 'Judging:|Label distribution|Judged|%'" || echo "  (No log output yet)"

# GPU utilization
echo ""
echo "[4] GPU Status:"
ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits" | awk '{printf "  GPU Util: %s%% | Memory: %s / %s MiB\n", $1, $2, $3}'

echo ""
echo "To see full log: ssh lambda 'tail -f ~/safety-refusal-stability/logs/judge_all.log'"
