#!/bin/bash
# Check progress of experiments running on Lambda

source .env

echo "=================================="
echo "Experiment Progress on Lambda H100"
echo "=================================="
echo ""

# Check running experiments
echo "[1] Running Experiments:"
ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "ps aux | grep 'python.*experiments' | grep -v grep | awk '{print \"  \" \$12, \$13, \$14, \$15}'" || echo "  None running"

# Count generation files
echo ""
echo "[2] Generation Files:"
LLAMA_COUNT=$(ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "ls ~/safety-refusal-stability/data/results/generations/*Llama*.csv 2>/dev/null | wc -l" | tr -d ' ')
MISTRAL_COUNT=$(ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "ls ~/safety-refusal-stability/data/results/generations/*mistral*.csv 2>/dev/null | wc -l" | tr -d ' ')
QWEN_COUNT=$(ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "ls ~/safety-refusal-stability/data/results/generations/*Qwen*.csv 2>/dev/null | wc -l" | tr -d ' ')
echo "  Llama: $LLAMA_COUNT / 20 files"
echo "  Mistral: $MISTRAL_COUNT / 20 files"
echo "  Qwen: $QWEN_COUNT / 20 files"

# Show recent log output
echo ""
echo "[3] Recent Log Activity:"
for model in mistral qwen; do
    LOG_FILE="~/safety-refusal-stability/logs/${model}_experiments.log"
    if ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "[ -f $LOG_FILE ]"; then
        echo "  ${model^}:"
        ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "tail -5 $LOG_FILE" | sed 's/^/    /'
    fi
done

# GPU status
echo ""
echo "[4] GPU Status:"
ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits" | awk '{printf "  GPU: %s%% util | %s/%s MiB | %s°C\n", $1, $2, $3, $4}'

echo ""
