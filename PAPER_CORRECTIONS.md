# Paper / repo corrections punch-list

Action items that need **your** decision or a re-run before this work is
submission-ready. The code/README fixes already applied are listed at the bottom;
everything in "Open" is left for you because it changes published numbers or
requires knowledge of what was actually run.

## Open — needs your decision

### 1. CONFIRM the dataset identity (highest priority)
The README and (likely) the paper describe the dataset as **BeaverTails**
(`PKU-Alignment/BeaverTails`, 330k_train, is_safe=False, "first 876 unique"),
citing Ji et al. 2023. But the **reproducible artifacts say otherwise**:

- `data/processed/prompts.csv` `source` column = **491 AdvBench + 385 HarmBench** (= 876)
- `src/data/load_datasets.py` only implements `load_advbench` / `load_harmbench` / `load_jailbreakbench` — there is **no BeaverTails loader**
- `data/raw/manifest.json` records only AdvBench + HarmBench downloads

I've corrected the **README** to match the artifacts (AdvBench+HarmBench). You must
decide which is ground truth and fix the other:
- If the experiments really ran on AdvBench+HarmBench → **correct the paper's
  dataset section + the Ji et al. citation.**
- If BeaverTails was actually intended → the committed `prompts.csv`,
  `load_datasets.py`, and `manifest.json` are wrong and must be regenerated.
- The old README also listed a harm-category breakdown (Copyright 12.8%, …). The
  `harm_category` column is empty in the processed data, so that breakdown is not
  regenerable; I removed it rather than reprint unverifiable numbers.

### 2. Table 2 %-Unstable (already in your VALIDATION_REPORT.md)
Update the paper's Table 2:
- Qwen 3 8B: % Unstable **27.7% → 11.8%**
- Gemma 3 12B: % Unstable **18.4% → 6.7%**

These strengthen the findings (Gemma is more stable than reported). Check any
figure that plots %-Unstable too.

### 3. Re-run the analysis scripts I just fixed, then update the paper's CIs
I seeded the unseeded bootstraps and removed the AUC data-leak (see below). The
**committed CIs/tables predate these fixes** and must be regenerated:
- `scripts/21_compute_auc_ci.py` — the AUC CI will **change** (the leak inflated
  it). Re-run and update the paper's AUC interval.
- `scripts/12_compare_judges.py` — inter-judge agreement CI (κ / 89% agreement) is
  now reproducible; re-run and lock the reported interval.
- `scripts/12_v2_comprehensive_analysis.py` — SSI-difference and correlation CIs.

### 4. Methodology decisions (not changed — they'd alter the metric definition)
- **temp=0.0 redundancy:** the 5 seeds at temperature 0 produce identical greedy
  outputs, so a temp-0 SSI of ~1.0 / flip rate ~0 is structural, not measured
  (`src/generation/vllm_runner.py`). Decide whether to drop the temp-0 duplicates
  or explicitly document this.
- **SSI denominator drops ERROR/UNKNOWN labels** (`09_within_temperature_analysis.py`,
  `demo/prepare_data.py`): a prompt with 18 ERROR + 2 REFUSE scores SSI=1.0.
  Decide how errored samples should count.
- **Hand-rolled Cohen's κ** in `scripts/11_sensitivity_and_confusion.py` can diverge
  from `sklearn.metrics.cohen_kappa_score` (already used in `12_compare_judges.py`).
  Recommend switching to sklearn and re-verifying the paper's κ = 0.62.

### 5. Repo hygiene (public repo)
- ~172 MB under `data/` + `artifact2_pilot_results.tar.gz` (3.3 MB) are tracked in
  a public repo → move to a GitHub Release / Git-LFS / `.gitignore`.
- `VALIDATION_REPORT.md` is currently untracked — decide whether to commit it (it
  documents the Table 2 fix) or keep it as internal notes.

## Already fixed (code + README)
- **README dataset section** rewritten to AdvBench+HarmBench (was BeaverTails);
  fabricated category breakdown removed; judge description clarified (Claude 3.5
  Haiku primary, Llama-3.1-70B cross-check); Quick Start notes that reproducing
  the reported numbers requires `13_batch_judge_anthropic.py`.
- **`scripts/21_compute_auc_ci.py`** — removed the bootstrap data-leak (was
  re-running 5-fold CV on with-replacement resamples, letting duplicated rows sit
  in both train and test folds); now bootstraps the honest held-out CV predictions.
- **`scripts/12_compare_judges.py`**, **`scripts/12_v2_comprehensive_analysis.py`** —
  seeded all bootstrap functions (`compute_agreement_ci`,
  `fixed_temp_sample_size_analysis`, `bootstrap_ssi_difference`,
  `correlation_with_ci`) so reported CIs are reproducible.
- **`requirements.txt`** — added `openai` (imported by `11_batch_judge_openai.py`).
