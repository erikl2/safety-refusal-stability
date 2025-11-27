#!/bin/bash
# Sync results FROM Lambda H100 back to local

set -e

source .env

echo "=================================="
echo "Syncing from Lambda H100"
echo "=================================="
echo "Host: $LAMBDA_HOST"
echo ""

# Sync labels back
echo "[1/2] Syncing judge labels..."
mkdir -p ./data/results/labels
rsync -avz --progress \
    -e "ssh -i $LAMBDA_SSH_KEY" \
    "$LAMBDA_HOST:~/safety-refusal-stability/data/results/labels/" \
    ./data/results/labels/

# Sync any new generations (if running experiments)
echo "[2/2] Syncing generations (if any new ones)..."
rsync -avz --progress \
    -e "ssh -i $LAMBDA_SSH_KEY" \
    "$LAMBDA_HOST:~/safety-refusal-stability/data/results/generations/" \
    ./data/results/generations/

echo ""
echo "Sync complete!"
echo ""
echo "Files downloaded:"
ls -lh ./data/results/labels/ | tail -n +2 | wc -l | xargs echo "  Labels:"
echo ""
echo "Next: Compute metrics with python scripts/06_compute_metrics.py"
