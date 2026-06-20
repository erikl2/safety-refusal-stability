#!/bin/bash
# Run judge on Lambda H100 instance
# This script runs judge on all generation files one at a time for fault tolerance

set -e  # Exit on error

# Load environment
source .env

echo "=================================="
echo "Running Judge on Lambda H100"
echo "=================================="
echo "Host: $LAMBDA_HOST"
echo ""

# Remote directory
REMOTE_DIR="~/safety-refusal-stability"

# SSH helper function
run_remote() {
    ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "$@"
}

# Check if venv exists on remote
echo "[1/4] Checking remote environment..."
if run_remote "[ ! -d $REMOTE_DIR/venv ]"; then
    echo "Creating virtual environment on remote..."
    run_remote "cd $REMOTE_DIR && python3 -m venv venv"
    echo "Installing dependencies..."
    run_remote "cd $REMOTE_DIR && source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt"
fi

# Ensure HF token is set
echo "[2/4] Setting up HuggingFace token..."
run_remote "export HF_TOKEN='$HF_TOKEN' && echo 'HF_TOKEN set'"

# Get list of generation files to judge
echo "[3/4] Finding generation files..."
FILES=$(run_remote "ls $REMOTE_DIR/data/results/generations/*.csv 2>/dev/null || true")

if [ -z "$FILES" ]; then
    echo "No generation files found! Please sync data first."
    exit 1
fi

FILE_COUNT=$(echo "$FILES" | wc -l | tr -d ' ')
echo "Found $FILE_COUNT generation files to judge"
echo ""

# Run judge on each file
echo "[4/4] Running judge..."
echo "This will take several hours on H100..."
echo ""

run_remote "cd $REMOTE_DIR && source venv/bin/activate && export HF_TOKEN='$HF_TOKEN' && cat > run_judge_loop.sh << 'SCRIPT_EOF'
#!/bin/bash
set -e

# Counter
total=\$(ls data/results/generations/*.csv 2>/dev/null | wc -l)
current=0

for genfile in data/results/generations/*.csv; do
    current=\$((current + 1))

    # Check if already labeled
    basename=\$(basename \"\$genfile\" .csv)
    labelfile=\"data/results/labels/\${basename}_labels.csv\"

    if [ -f \"\$labelfile\" ]; then
        echo \"[\$current/\$total] SKIP: \$basename (already labeled)\"
        continue
    fi

    echo \"\"
    echo \"========================================\"
    echo \"[\$current/\$total] Judging: \$basename\"
    echo \"========================================\"

    # Run judge with error handling
    if python scripts/05_run_judge.py --file \"\$genfile\" --batch-size 16; then
        echo \"SUCCESS: \$basename\"
    else
        echo \"ERROR: Failed on \$basename (continuing to next file)\"
        echo \"\$genfile\" >> failed_files.txt
    fi
done

echo \"\"
echo \"Judging complete!\"
if [ -f failed_files.txt ]; then
    echo \"WARNING: Some files failed. See failed_files.txt\"
    cat failed_files.txt
fi
SCRIPT_EOF
chmod +x run_judge_loop.sh && ./run_judge_loop.sh"

echo ""
echo "Judge run complete!"
echo "Results saved to data/results/labels/ on remote"
echo ""
echo "Next steps:"
echo "  1. Sync results: ./scripts/sync_from_lambda.sh"
echo "  2. Compute metrics: python scripts/06_compute_metrics.py"
