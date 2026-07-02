#!/usr/bin/env python3
"""23_build_public_export.py — stage the clean-publish export (labels-only, fresh history).

Builds a publishable copy of this repo at
``~/code_workspace/publish_staging/safety-refusal-stability-public/`` containing code,
paper source, and LABELS-ONLY artifacts — never raw model responses, never judge
batch payloads, and no raw prompt text (prompts are regenerable from the public
AdvBench/HarmBench sources via scripts/01-02). The original repo is untouched.

Content-minimization rules encoded here:
  - label CSVs that carry raw text columns (prompt/response/reasoning) are STRIPPED
    to (prompt_id, label, model, temperature, seed);
  - per_prompt_metrics.csv drops its ``prompt`` column;
  - prompts.csv drops its ``prompt`` column (id/source/lengths kept — enough to join
    every published label to the public benchmark row);
  - responses/, batch_jobs/, generations/, raw benchmark CSVs: excluded entirely.

After staging, verify with thinking-dangerously/scripts/scan_export.py (the
content-min fingerprint scan) and run scripts/22_recompute_paper_numbers.py inside
the export to prove the paper's numbers recompute from what ships.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO.parent / "publish_staging" / "safety-refusal-stability-public"

# Verbatim copies: code, configs, paper source + referenced figures.
VERBATIM_DIRS = ["src", "configs"]
PAPER_FIGS = [
    "flip_rate.pdf", "ssi_distribution.pdf", "temperature_effect.pdf",
    "stability_vs_compliance.pdf", "multimodel_stability.pdf",
    "multimodel_distribution.pdf", "sample_size_reliability.pdf",
]
# Scripts ship (they are the recompute path); shell helpers for Lambda too — code, not data.
SCRIPT_GLOBS = ["*.py", "*.sh"]

# Labels-only CSVs that ship verbatim (schema: model,prompt_id,temperature,seed,label,judge).
LABELS_VERBATIM = [
    "claude_haiku_llama_qwen25.csv",
    "claude_haiku_new_models.csv",
    "gpt4o_mini_new_models.csv",
]
# Per-config judge CSVs that carry raw text columns -> strip to these columns.
STRIP_KEEP = ["prompt_id", "label", "model", "temperature", "seed"]
# Mistral files are excluded outright: not part of the paper's four models.
STRIP_GLOBS = ["meta_llama_*_labels.csv", "Qwen_Qwen2.5_*_labels.csv"]

README = """\
# Safety Refusal Stability — public release (labels + recompute)

Code and labels-only artifacts for *"The Instability of Safety: How Random Seeds and
Temperature Expose Inconsistent LLM Refusal Behavior"* (paper source in `paper/`).

Four instruction-tuned models (Llama 3.1 8B, Qwen 2.5 7B, Qwen 3 8B, Gemma 3 12B),
876 harmful prompts (491 AdvBench + 385 HarmBench), 20 sampling configurations
(4 temperatures x 5 seeds), 70,080 generations.

## What is (and is not) in this repo — content minimization

This release contains **no model responses and no prompt text**. Safety-refusal runs
produce raw completions to harmful prompts; we do not redistribute those. What ships:

- `data/results/labels/` — per-(model, prompt_id, temperature, seed) judge labels
  (REFUSE / PARTIAL / COMPLY). These are the exact inputs to every number in the paper.
- `data/results/metrics/` — derived aggregates (SSI, flip rates, CIs, test statistics).
- `data/processed/prompts.csv` — prompt **index** (id, source benchmark, length) with
  the text column removed. Rebuild the full file from the public benchmarks:
  `python scripts/01_download_data.py && python scripts/02_preprocess.py`
  (AdvBench and HarmBench are MIT-licensed public datasets; ids are stable).
- `paper/` — LaTeX source + figures (v3: corrected dataset attribution + CIs).

## Reproduce the paper's numbers from the shipped labels

    python scripts/22_recompute_paper_numbers.py

reproduces Table 2 and the section 4.1-4.3 statistics directly from
`data/results/labels/`. To re-run the full pipeline (generation + judging) from
scratch, see `scripts/01`-`13` (requires GPUs / API keys; regenerates the raw data
this release deliberately omits).

## Citation

See `paper/main.tex` (arXiv version forthcoming).
"""

DATA_README = """\
# data/ — labels-only release

- `processed/prompts.csv`: prompt index (id, source, char_length, word_count). The
  `prompt` text column is deliberately removed; regenerate via scripts/01-02 from the
  public AdvBench/HarmBench sources (ids are deterministic).
- `results/labels/*.csv`: judge labels. The per-config `*_labels.csv` files are
  stripped to (prompt_id, label, model, temperature, seed); the `claude_haiku_*` /
  `gpt4o_mini_*` files were labels-only at creation.
- `results/metrics/`: derived aggregates only. `per_prompt_metrics.csv` ships without
  its prompt-text column.
- NOT included: `results/responses/`, `results/batch_jobs/`, `results/generations/`,
  `raw/` benchmark dumps — raw completions to harmful prompts are not redistributed.
"""


def strip_csv(src: Path, dst: Path, keep: list[str]) -> None:
    with open(src, newline="") as fin, open(dst, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        cols = [c for c in keep if c in (reader.fieldnames or [])]
        writer = csv.DictWriter(fout, fieldnames=cols)
        writer.writeheader()
        for row in reader:
            writer.writerow({c: row[c] for c in cols})


def main() -> int:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    (DEST / "README.md").write_text(README)
    shutil.copy2(REPO / "requirements.txt", DEST / "requirements.txt")

    for d in VERBATIM_DIRS:
        shutil.copytree(REPO / d, DEST / d, ignore=shutil.ignore_patterns("__pycache__"))

    sdst = DEST / "scripts"
    sdst.mkdir()
    for pat in SCRIPT_GLOBS:
        for f in sorted((REPO / "scripts").glob(pat)):
            shutil.copy2(f, sdst / f.name)

    pdst = DEST / "paper" / "figures"
    pdst.mkdir(parents=True)
    for name in ("main.tex", "main.pdf"):
        shutil.copy2(REPO / "paper" / name, DEST / "paper" / name)
    for fig in PAPER_FIGS:
        shutil.copy2(REPO / "paper" / "figures" / fig, pdst / fig)

    ddst = DEST / "data"
    (ddst / "processed").mkdir(parents=True)
    (ddst / "results" / "labels").mkdir(parents=True)
    (ddst / "README.md").write_text(DATA_README)

    # prompt index: drop the text column (and the empty harm_category column).
    strip_csv(REPO / "data/processed/prompts.csv", ddst / "processed/prompts.csv",
              keep=["id", "source", "char_length", "word_count"])

    labels_src = REPO / "data/results/labels"
    for name in LABELS_VERBATIM:
        shutil.copy2(labels_src / name, ddst / "results/labels" / name)
    n_stripped = 0
    for pat in STRIP_GLOBS:
        for f in sorted(labels_src.glob(pat)):
            strip_csv(f, ddst / "results/labels" / f.name, keep=STRIP_KEEP)
            n_stripped += 1

    mdst = ddst / "results" / "metrics"
    mdst.mkdir()
    for f in sorted((REPO / "data/results/metrics").glob("*.csv")):
        if f.name == "per_prompt_metrics.csv":
            strip_csv(f, mdst / f.name, keep=[
                "prompt_id", "refusal_rate", "partial_rate", "comply_rate",
                "majority_label", "stability_index", "flip_occurred", "num_samples"])
        else:
            shutil.copy2(f, mdst / f.name)

    (DEST / ".gitignore").write_text("__pycache__/\n*.pyc\n.env\n")

    # Fresh history: a brand-new repo whose first commit contains only what ships.
    subprocess.run(["git", "init", "-q"], cwd=DEST, check=True)
    subprocess.run(["git", "add", "-A"], cwd=DEST, check=True)
    subprocess.run(["git", "commit", "-q", "-m",
                    "Public release: code + labels-only artifacts + paper v3"],
                   cwd=DEST, check=True)

    print(f"staged: {DEST}")
    print(f"label CSVs: {len(LABELS_VERBATIM)} verbatim, {n_stripped} stripped to {STRIP_KEEP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
