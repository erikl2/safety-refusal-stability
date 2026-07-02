# arXiv v3 — exact submit steps (human step, gate G8)

> Prepared 2026-07-01 by the autopilot (SHIP-PREP session). Mechanical prep only —
> the submission click is yours.

## The bundle

**File:** `paper/arxiv_submission_v3.zip`
**SHA256:** `a840de6660b8f9fff53609e6494ca6434a04200d6b0be8f7e08f0254b89412e7`
**Contents (9 files):** `main.tex` (corrected v3, 49,780 bytes) + the 7 figure PDFs
actually referenced by `\includegraphics` (flip_rate, ssi_distribution,
temperature_effect, stability_vs_compliance, multimodel_stability,
multimodel_distribution, sample_size_reliability).

**Verified 2026-07-01:** compiles clean with `pdflatex` (TeX Live 2024), two passes,
**zero errors, zero undefined references/citations**; only benign `h→ht` float
warnings. Output PDF ≈ 544 KB, byte-consistent with `paper/main.pdf`.

**Deliberately EXCLUDED from the bundle (do not re-add):**
- `main.bbl` — **not used**: the bibliography is inline (`\begin{thebibliography}`,
  main.tex:455) and covers all 16 cited keys, verified key-by-key. No BibTeX run occurs.
- `references.bib` — same reason; arXiv doesn't need it.
- `neurips_2024.sty` — a 14-byte "404: Not Found" download stub, never `\usepackage`d.
- `.png` duplicates + 3 unreferenced figures (model_comparison,
  within_temperature_stability, sample_size_reliability_temp07).
- The old `arxiv_submission.zip` (Dec 12 tex — pre-correction; superseded, kept for provenance).

**What v3 fixes vs what's on arXiv** (see `PAPER_CORRECTIONS.md` for the full audit):
dataset attribution corrected to AdvBench(491)+HarmBench(385) (was misattributed to
BeaverTails), Table-2/body label-mismatch digit fixes, Wilson CIs + paired temperature CI,
judge-attribution map pinned from labels, keyword-derived category caveat at §4.5.

## Submit steps (~10 min)

1. Log in at https://arxiv.org/user/ (account under elarsen.mailbox@gmail.com).
2. Find the existing submission and click **Replace**.
   ⚠️ The arXiv ID is not recorded anywhere in this repo — grab it from your arXiv
   account page, then add it to README.md while you're at it (one line: `arXiv:XXXX.XXXXX`).
   (If the paper was never actually posted, this becomes **Start New Submission** instead;
   license: arXiv's default perpetual non-exclusive is fine, same as v1.)
3. Upload `paper/arxiv_submission_v3.zip`. arXiv's AutoTeX will run pdflatex — it
   succeeds locally with zero unresolved refs, so expect green.
4. **Update the metadata abstract** — the abstract changed in v3 (89.1% agreement /
   κ=0.62 / corrected percentages). Copy it fresh from main.tex (the `\begin{abstract}`
   block) into the metadata field; strip the `\textbf{}`/math markup arXiv rejects.
5. In "Comments," add one line, e.g.:
   `v3: corrects dataset attribution (AdvBench+HarmBench, not BeaverTails), fixes
   label-derived statistics and CIs; conclusions unchanged.`
   (Honest-correction norm: name what changed, don't bury it.)
6. Preview the AutoTeX-built PDF, spot-check Table 2 (Qwen3 11.8 / Gemma 6.7 %-Unstable)
   and §3 Dataset (AdvBench+HarmBench), then **Submit**.

## After it's announced

- Record the new version URL in README.md and PIPELINE_STATUS.md.
- The clean-publish repo export (separate gate) can then link the arXiv v3.
