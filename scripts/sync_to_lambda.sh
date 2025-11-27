#!/bin/bash
# Sync code and data TO Lambda H100

set -e

source .env

echo "=================================="
echo "Syncing to Lambda H100"
echo "=================================="
echo "Host: $LAMBDA_HOST"
echo ""

# Create remote directory if it doesn't exist
ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "mkdir -p ~/safety-refusal-stability"

# Sync code (exclude data, venv, cache)
echo "[1/3] Syncing code..."
rsync -avz --progress \
    -e "ssh -i $LAMBDA_SSH_KEY" \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude 'data/results/labels/' \
    --exclude 'paper/' \
    --exclude 'notebooks/' \
    ./ "$LAMBDA_HOST:~/safety-refusal-stability/"

# Sync generation data (if not already there)
echo "[2/3] Syncing generation data..."
rsync -avz --progress \
    -e "ssh -i $LAMBDA_SSH_KEY" \
    ./data/results/generations/ \
    "$LAMBDA_HOST:~/safety-refusal-stability/data/results/generations/"

# Create labels directory on remote
echo "[3/3] Creating labels directory..."
ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "mkdir -p ~/safety-refusal-stability/data/results/labels"

echo ""
echo "Sync complete!"
echo ""
echo "Next: Run judge with ./scripts/run_judge_on_lambda.sh"
