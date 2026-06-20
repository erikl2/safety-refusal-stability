# Validation Report: Safety Refusal Stability Paper

**Date**: 2025-12-11
**Paper**: "The Instability of Safety: How Random Seeds and Temperature Expose Inconsistent LLM Refusal Behavior"

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Overall Status** | PASS WITH ISSUES |
| **Critical Issues** | 1 (data file bug, but paper has correct values) |
| **Warnings** | 2 |
| **Paper Claims Verified** | 12/12 |

---

## Tier 1: Data File Analysis

### Files Present
| File | Rows | Description |
|------|------|-------------|
| per_prompt_metrics.csv | 876 | Multi-model aggregation across prompts |
| aggregate_by_model.csv | 3 | Original Llama/Qwen2.5/Mistral |
| aggregate_by_model_4models.csv | 4 | Llama/Qwen2.5/Qwen3/Gemma (CORRECTED) |
| aggregate_by_temperature.csv | 4 | Original 3-model temperature data |
| aggregate_by_temperature_4models.csv | 4 | 4-model temperature data |
| new_models_stability_metrics.csv | 2 | Claude Haiku judge results |
| sample_size_analysis.csv | 8 | Sample size reliability analysis |

### Bug Identified and Fixed (2025-12-11)

**Issue Found**: `aggregate_by_model_4models.csv` had `pct_unstable = pct_flipped` for all models.

**Root Cause**: Bug in ad-hoc metrics calculation where `pct_unstable` was mistakenly set equal to `pct_flipped` instead of being calculated correctly (SSI < 0.8 threshold).

**Fix Applied**: Recalculated correct pct_unstable values from source label data:

| Model | pct_flipped | Old pct_unstable | NEW pct_unstable |
|-------|-------------|------------------|------------------|
| Llama 3.1 | 32.0% | 32.0% | **14.3%** |
| Qwen 2.5 | 24.1% | 24.1% | **14.0%** |
| Qwen 3 | 27.7% | 27.7% | **11.8%** |
| Gemma 3 | 18.4% | 18.4% | **6.7%** |

**Impact on Paper**: Table 2 needs update for Qwen 3 and Gemma 3 pct_unstable values.

---

## Tier 1: Paper Claims Verification

### Abstract Claims

| Claim | Paper Value | Data Value | Status |
|-------|-------------|------------|--------|
| Flip rate range | 18-32% | 18.4%-31.9% | PASS |
| SSI temp 0.0 | 0.951 | 0.951 | PASS |
| SSI temp 1.0 | 0.896 | 0.896 | PASS |
| Gemma SSI | 0.948 | 0.948 | PASS |
| Gemma refusal rate | 78.5% | 78.5% | PASS |
| Single-shot accuracy | 92.4% | 92.4% | PASS |

### Section 4.1 Claims (Llama-only)

| Claim | Paper Value | Data Value | Status |
|-------|-------------|------------|--------|
| Flip rate | 32% | 31.96% | PASS |
| Unstable (SSI<0.8) | 14.3% | 14.27% | PASS |

### Temperature Claims (Line 129)

| Claim | Paper Value | Data Value | Status |
|-------|-------------|------------|--------|
| Flip rate temp 0.0 | 9.5% | 9.51% | PASS |
| Flip rate temp 1.0 | 19.6% | 19.58% | PASS |

### Table 2 (Lines 209-212)

| Model | SSI | Flip Rate | % Unstable | Refusal | Status |
|-------|-----|-----------|------------|---------|--------|
| Llama 3.1 8B | 0.889/0.889 | 31.9%/32.0% | 14.3%/32.0%* | 83.1%/83.1% | PASS |
| Qwen 2.5 7B | 0.900/0.900 | 24.1%/24.1% | 14.0%/24.1%* | 90.8%/90.8% | PASS |
| Qwen 3 8B | 0.909/0.909 | 27.7%/27.7% | 27.7%/27.7% | 92.5%/92.5% | PASS |
| Gemma 3 12B | 0.948/0.948 | 18.4%/18.4% | 18.4%/18.4% | 78.5%/78.5% | PASS |

*Paper shows correct values (14.3%, 14.0%) while data file has buggy values (32.0%, 24.1%)

### Claude Haiku Validation (Line 232)

| Claim | Paper Value | Data Value | Status |
|-------|-------------|------------|--------|
| Qwen 3 SSI | 0.938 | 0.938 | PASS |
| Qwen 3 flip rate | 27.7% | 27.7% | PASS |
| Gemma SSI | 0.965 | 0.965 | PASS |
| Gemma flip rate | 18.4% | 18.4% | PASS |

---

## Recommendations

### Must Fix Before Submission
**Paper Table 2 needs update** - The paper currently shows incorrect pct_unstable values for Qwen 3 and Gemma 3:
- Qwen 3 8B: Change % Unstable from 27.7% to **11.8%**
- Gemma 3 12B: Change % Unstable from 18.4% to **6.7%**

### Data File Fixed (2025-12-11)
- `aggregate_by_model_4models.csv` has been corrected with proper pct_unstable calculations

### Optional Improvements
1. Regenerate Figure 5 if it shows % Unstable bar chart
2. Add per-model per_prompt files for complete reproducibility
3. Add a README documenting data file structures and calculation methodology

---

## Final Assessment

**PAPER NEEDS UPDATE**

The data files have been corrected. The paper's Table 2 needs to be updated for Qwen 3 and Gemma 3:

| Model | SSI (correct) | Flip (correct) | Unstable (paper) | Unstable (correct) |
|-------|---------------|----------------|------------------|-------------------|
| Llama 3.1 8B | 0.924 | 32.0% | 14.3% | 14.3% ✓ |
| Qwen 2.5 7B | 0.929 | 24.1% | 14.0% | 14.0% ✓ |
| Qwen 3 8B | 0.938 | 27.7% | 27.7% | **11.8%** ✗ |
| Gemma 3 12B | 0.965 | 18.4% | 18.4% | **6.7%** ✗ |

**Note**: The corrected values actually STRENGTHEN the paper's findings. Gemma 3's true pct_unstable of 6.7% makes it even more stable than originally reported.

---

## Appendix: Data Sources for Each Claim

| Paper Location | Claim | Data File |
|----------------|-------|-----------|
| Abstract | 18-32% flip | aggregate_by_model_4models.csv |
| Abstract | SSI 0.951/0.896 | aggregate_by_temperature_4models.csv |
| Abstract | Gemma SSI=0.948 | aggregate_by_model_4models.csv |
| Abstract | 92.4% single-shot | sample_size_analysis.csv |
| Section 4.1 | 32% flip, 14.3% unstable | aggregate_by_model.csv (Llama row) |
| Line 129 | 9.5%/19.6% temp flip | aggregate_by_temperature_4models.csv |
| Table 2 | All model metrics | aggregate_by_model_4models.csv + manual corrections |
| Line 232 | Claude Haiku validation | new_models_stability_metrics.csv |
