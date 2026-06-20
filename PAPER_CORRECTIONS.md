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

> **✅ RESOLVED 2026-06-19 — dataset identity verified = AdvBench(491) + HarmBench(385) = 876.**
> v1's SSI metric, recomputed from the label artifacts (label column only), reproduces the paper's
> reported numbers EXACTLY — Table 2 all four models to three decimals, and §4.1–4.3 Llama (32% flip,
> 14.3% unstable). Every label file's 876 prompt_ids join to `prompts.csv` with **0 orphans** and
> exactly **491 advbench + 385 harmbench**. So the "BeaverTails / 330k_train / is_safe=False" text was
> **stale, not a different run**. `paper/main.tex` dataset attribution corrected (intro, §3 Dataset,
> §6 Dataset-Scope limitation, dataset citation → AdvBench~\cite{zou2023advbench} +
> HarmBench~\cite{mazeika2024harmbench}). The `ji2023beavertails` bibitem is **kept** — related-work §2
> still cites BeaverTails legitimately as a benchmark, so deleting it would orphan that citation.
> **Harm-category breakdown is keyword-derived:** because `harm_category` is empty, the paper's category
> analysis (incl. the copyright cluster, N=112) is assigned post-hoc by keyword matching on prompt text —
> now stated explicitly at §4.5 so the copyright-cluster claim is transparently sourced.

### 2. Table 2 %-Unstable (already in your VALIDATION_REPORT.md)
Update the paper's Table 2:
- Qwen 3 8B: % Unstable **27.7% → 11.8%**
- Gemma 3 12B: % Unstable **18.4% → 6.7%**

These strengthen the findings (Gemma is more stable than reported). Check any
figure that plots %-Unstable too.

> **✅ RESOLVED 2026-06-19 (v3) — verified cell-by-cell against the labels.** The %-Unstable
> corrections were ALREADY in Table 2 (Qwen3 11.8, Gemma 6.7) and reproduce exactly from the
> Claude-judge labels — this note was stale. Remaining label-mismatches found & fixed in main.tex:
> Gemma flip 18.3→**18.4**, Qwen3 flip 27.8→**27.7**, Gemma refusal 78.6→**78.5**, Llama refusal
> 79.2→**79.3** (the last two also reconcile Table 2 with the §4.4/§4.5 body text), and §1's
> "24.8%"→**24.9%** (matches §4.4 + labels). `tab:within_temp`, `tab:binary_ablation`, and
> `tab:threshold_sensitivity` match the labels exactly — unchanged. The %-Unstable FIGURE
> (`model_comparison.{png,pdf}`) is a binary image I can't recompute — eyeball it for 6.7/10.4/11.8/12.0.

### 3. Re-run the analysis scripts I just fixed, then update the paper's CIs
I seeded the unseeded bootstraps and removed the AUC data-leak (see below). The
**committed CIs/tables predate these fixes** and must be regenerated:
- `scripts/21_compute_auc_ci.py` — the AUC CI will **change** (the leak inflated
  it). Re-run and update the paper's AUC interval.
- `scripts/12_compare_judges.py` — inter-judge agreement CI (κ / 89% agreement) is
  now reproducible; re-run and lock the reported interval.
- `scripts/12_v2_comprehensive_analysis.py` — SSI-difference and correlation CIs.

> **✅ RESOLVED 2026-06-19 (v3) — CIs regenerated from the labels.** Flip-rate and %-Unstable CIs in
> Table 2 and §4.1 replaced with **Wilson** score intervals (correct for a single proportion over 876
> prompts); the temperature-decrease CI (§4.3, 0.035) regenerated as a **paired** CI over
> per-(model,prompt) seed-level SSI differences → **[0.031, 0.039]**. Mean-SSI CIs are bootstrap
> (verified to agree within ±0.001; left as-is). Table 2 caption now names the methods. Inter-judge
> agreement reproduces (89.0% under the 35,040-pair denominator that counts non-VALID as disagreement;
> 89.1% on the 35,005 valid pairs), κ=0.62; Spearman ρ −0.47→−0.70 reproduces exactly.
> **NOT regenerated (content-minimization):** the AUC CI [0.69,0.78] and the linguistic-feature
> classifier require reading raw prompt text — out of bounds this pass; left for a local run.

### 6. Judge attribution (v3)
> **✅ RESOLVED 2026-06-19 (v3) — judge map pinned from the labels, not guessed.** Recomputing every
> §4.1-4.3 number under BOTH judges' label files (`scripts/22_recompute_paper_numbers.py`) shows the
> split is at §4.3, not §4.4: **§4.1-4.2 (single-model Llama deep-dive) reproduce ONLY from the
> Llama-70B judge** (flip 32%, %unstable 14.3%, 83.1/10.9/5.9 — Claude-judge gives 27.3 / 10.4 /
> 79.3-12.8-7.9), while **§4.3 + the within-temperature table reproduce ONLY from the Claude unified
> judge** (within-temp flip 5.1→23.6, not the Llama-judge's 5.4→28.9; and the cross-model mean
> 0.977→0.942 is impossible under the Llama judge, which only labelled 3 of the 4 models). The old
> Judge Note ("Sections 4.1--4.3 … Llama 70B judge") mis-stated §4.3's source. **Fixed:** Judge Note
> now says §4.1-4.2 = Llama-70B, §4.3 + within-temp table + Table 2 = Claude unified judge; §4.3 opener
> and the within-temp caption state the judge explicitly. Numbers unchanged (they were correct).

### 7. Judge×model coverage + unreported models/judges (v3)
> **✅ AUDITED 2026-06-19 (v3) — pinned from the label files (`scripts/22_recompute_paper_numbers.py`).**
> Judge×model coverage (all per-config, 17,520 rows = 876×20 each):
> - **Claude 3.5 Haiku** judged **all four reported models** (Llama-8B, Qwen2.5, Qwen3, Gemma; judge col
>   `claude-3-5-haiku-20241022`). So §3's "Claude only for the newer models (Qwen3/Gemma)" was WRONG and
>   contradicted §4.6 ("all 70,080 responses using Claude"). **Fixed:** §3 Response Classification + the
>   Limitations Judge-Model note now say Llama-70B judges the single-model Llama deep-dive (§4.1-4.2) and
>   Claude is the unified judge across all four for cross-model results (§4.3, Table 2).
> - **Llama-3.1-70B** judged Llama-8B, Qwen2.5, **Mistral-7B-v0.3** only — NOT Qwen3/Gemma, so it is not a
>   "primary judge across all models."
> - **GPT-4o-mini** judged Qwen3 + Gemma only — a 3rd judge the paper never mentions.
> **Does any unreported artifact feed a reported number? NO (verified by value):**
> - GPT-4o-mini → Qwen3 SSI .867/flip 45.8/unstable 28.7, Gemma .986/3.4/3.2 (Gemma only n=592): matches
>   NOTHING reported (Table 2 = Claude .938/.965); κ=0.62/89% is explicitly Llama-70B vs Claude on
>   Llama-8B+Qwen2.5. GPT-4o-mini feeds no reported number.
> - Mistral-7B-v0.3 (.748/78.9/55.5) is a 5th model in no reported table.
> - `metrics/aggregate_by_model_4models.csv` is a MIXED-judge file (Llama-judge rows for Llama-8B+Qwen2.5,
>   Claude-judge rows for Qwen3/Gemma) and is **read by no script** (`grep` over scripts/ + demo/). Table 2
>   and the multimodel figures use pure Claude (`unified_claude_haiku_all_models.csv` + the `claude_haiku_*`
>   labels). Dead/exploratory.
> **Conclusion:** Mistral, GPT-4o-mini, and the mixed file are purely exploratory and feed nothing reported;
> the paper is left as-is on that point. Whether to disclose the extra exploratory judge/model in a footnote
> is Erik's call.

### 8. GPT-5.5 foil v3 fix-pack — the 3 real items (label-recomputed via scripts/22, not hand-patched)
> **✅ RESOLVED 2026-06-20 (v3).** The decorrelated GPT-5.5 foil (`critiques/v3-paper_main_*.md`,
> `critiques/v3-paper_triage_*.md`) surfaced 3 real issues (11 needs-Erik, 0 hallucinated). All recomputed
> from the labels (`scripts/22` v3 FIX-PACK section), before→after:
> - **[2] Inter-judge confusion matrix denominator.** The 9 cells sum to **35,005** (label-verified exact:
>   27,933/1,580/940 · 197/1,842/558 · 6/532/1,417), but the Total row + caption said 35,040. Locked to the
>   35,005 VALID pairs (35 excluded where a judge returned a non-three-class label): Total 35,040→**35,005**;
>   agreement 89.0%→**89.1%** (31,192/35,005); κ stays **0.62** (0.621). Resolves **[8]**: κ covers ONLY
>   Llama-8B + Qwen2.5 — the two models with both judges (Llama-70B never judged Gemma/Qwen3) — now stated in
>   the caption, §4.6, §4.7, and the abstract.
> - **[12] "9.5%→19.6% instability" (§6, §7).** Traced to a stale `metrics/aggregate_by_temperature_4models.csv`
>   (pct_flipped 9.506→19.578) that does NOT match the displayed within-temperature table. Replaced with the
>   label-recomputed **mean within-temperature flip rate 7.8%→19.4%** (recompute 7.82→19.35; matches
>   `tab:within_temp` exactly), now cited to that table.
> - **[3] Sample-size 99.4%@N=20 vs 100%@full.** Fixed-temp table method confirmed without-replacement (B=500,
>   `12_v2_comprehensive_analysis.py`). The POOLED table's generator (`sample_size_analysis.csv`) is **not in
>   the repo**. Under the prose-described without-replacement method, N=20 draws all 20 configs ⇒ 100% by
>   construction: fixed the N=20 row 99.4→**100.0** and stated the method in §4.3 + the caption.
>   **OPEN flag for Erik:** a from-labels recompute (Claude unified judge, 4 models, no replacement) gives
>   N=1=**94.6%**, not the table's 92.4% — so the pooled column is a DIFFERENT population (likely the Llama-8B
>   Llama-judge deep-dive). The N<20 rows + the headline 92.4% should be regenerated with the population stated;
>   NOT changed here (would conflate populations).
>   **→ RESOLVED 2026-06-20 (Erik picked (a)).** Both populations recomputed (`scripts/22` (C)); the pooled
>   table IS **(a) Llama-3.1-8B single-model, Llama-70B judge** — N1/3/5/10/20 = 92.5/95.1/96.3/97.8/100,
>   matching the paper to ±0.1 (bootstrap-seed noise). Stated explicitly in the `tab:sample_size` caption,
>   §4.8 prose, and the abstract. **NEW surprise (verified):** the FIXED-TEMP table (94.2--97.7%) is
>   population **(b) all four models, Claude unified judge** (recompute t0=97.7/t1=94.2 EXACT under (b);
>   98.7/90.3 under (a)). So §4.8 MIXES populations, and §4.8.1 directly compares 97.7% (b) vs 92.4% (a)
>   across different populations. Left for Erik (a framing/re-analysis call beyond the clean pooled
>   attribution): scope the fixed-temp to (b), or re-run one analysis on the other's population.
>   The §3 footnote (Mistral-7B + GPT-4o-mini exploratory runs) is now inserted.
>   **→ RESOLVED 2026-06-20 (reliability story UNIFIED on (a)).** Erik approved rewriting the entire
>   reliability analysis on population (a) Llama-3.1-8B / Llama-70B judge (recompute: `scripts/22` (D),
>   without replacement, B=500). The fixed-temp table (`tab:fixed_temp_sample_size`) was replaced with the
>   (a) values — N=1 row **98.7/95.1/92.8/90.3** (t=0/0.3/0.7/1.0), N=3 **99.7/97.4/96.0/94.8**, N=5=**100**
>   (full seed set, by construction). §4.8.1, abstract, contributions (§1), §6, and §7 rewritten: pooled
>   single-shot **92.4→92.5%** (and table N=1/N=3 to 92.5/95.1); fixed-greedy **97.7→98.7%**; the flat
>   "N≥3 for >97%" replaced by the temperature-graded rule (>97% at: t=0 → N=1, t≤0.3 → N=3, t≥0.7 → full
>   5-seed set, pooled → N=10). All values label-verified. The mixed-population issue (pooled=a,
>   fixed-temp=b) is now gone — both are (a).

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
