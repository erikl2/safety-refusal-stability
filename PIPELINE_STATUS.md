# Pipeline Status - Safety Refusal Stability

## Current Status: ✅ Llama Complete | 🔄 Mistral + Qwen Running

### Completed: Llama 3.1 8B

**Results:**
- ✅ 20 generation files (17,520 responses)
- ✅ All responses judged and labeled
- ✅ Metrics computed
- ✅ Figures generated

**Key Findings:**
- **Mean SSI: 0.799** (borderline stable)
- **32% unstable prompts** (SSI < 0.8) - significant variance!
- **32% show flips** (behavior changes across seeds/temps)
- **Label distribution:** 83.1% REFUSE, 10.9% PARTIAL, 5.9% COMPLY

### In Progress: Mistral + Qwen

**Running on Lambda H100:**
```bash
# Mistral 7B Instruct (running now)
# Qwen 2.5 7B Instruct (queued, will start after Mistral)
```

**Estimated time:** 4-6 hours total

**Monitor progress:**
```bash
./scripts/check_experiment_progress.sh
```

**Check logs on Lambda:**
```bash
ssh lambda "tail -f ~/safety-refusal-stability/logs/all_experiments.log"
```

---

## Next Steps (When Experiments Complete)

### 1. Check if experiments finished
```bash
./scripts/check_experiment_progress.sh
# Should show: Mistral 20/20, Qwen 20/20
```

### 2. Sync generation results
```bash
./scripts/sync_from_lambda.sh
```

### 3. Judge all new generations
```bash
# On Lambda (run in background)
ssh lambda "cd ~/safety-refusal-stability && source venv/bin/activate && export HF_TOKEN='$HF_TOKEN' && nohup bash -c 'for f in data/results/generations/*mistral*.csv data/results/generations/*Qwen*.csv; do basename=\$(basename \"\$f\" .csv); labelfile=\"data/results/labels/\${basename}_labels.csv\"; [ -f \"\$labelfile\" ] && continue; python scripts/05_run_judge.py --file \"\$f\" --judge-model meta-llama/Llama-3.1-8B-Instruct --batch-size 32; done' > logs/judge_new_models.log 2>&1 &"

# Monitor:
./scripts/check_judge_progress.sh
```

### 4. Sync labels back
```bash
./scripts/sync_from_lambda.sh
```

### 5. Recompute metrics with all 3 models
```bash
python scripts/06_compute_metrics.py
```

### 6. Regenerate figures with comparisons
```bash
python scripts/07_generate_figures.py
```

---

## Quick Commands

**Check experiment progress:**
```bash
./scripts/check_experiment_progress.sh
```

**Check judge progress:**
```bash
./scripts/check_judge_progress.sh
```

**Sync everything from Lambda:**
```bash
./scripts/sync_from_lambda.sh
```

**View GPU status:**
```bash
ssh lambda "nvidia-smi"
```

**Kill all processes (emergency):**
```bash
ssh lambda "pkill -f python"
```

---

## Expected Final Results

When all 3 models are complete:
- **876 prompts × 20 configs × 3 models = 52,560 total responses**
- Full stability comparison across model families
- Per-model SSI scores
- Temperature effects by model
- Model × harm category heatmaps

---

**Last updated:** $(date)
