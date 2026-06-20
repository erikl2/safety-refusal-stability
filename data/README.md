# Data

**No prompt or response data is committed to this repository.** The harmful-prompt
benchmarks are red-teaming datasets; this repo fetches them from their official sources
at setup time rather than redistributing them, and all generated model responses /
labels are produced locally and stay local. This keeps the public repo free of harmful
prompts and raw model completions.

## Fetch the datasets

```bash
pip install -r requirements.txt
python scripts/01_download_data.py     # writes data/raw/{advbench,harmbench[,jailbreakbench]}.csv + manifest.json
python scripts/02_preprocess.py        # builds data/processed/prompts.csv (the 876-prompt set)
```

The study uses **876 harmful prompts = 491 AdvBench + 385 HarmBench** (the `source` column
of `data/processed/prompts.csv` is ground truth). `harm_category` is empty in the processed
set, so no category breakdown is derived from it.

## Sources, licenses & attribution

| Dataset | Official source (fetched by `01_download_data.py`) | License |
|---|---|---|
| **AdvBench** (491) | `llm-attacks/llm-attacks` → `data/advbench/harmful_behaviors.csv` | MIT (Zou et al. 2023) |
| **HarmBench** (385) | `centerforaisafety/HarmBench` → `data/behavior_datasets/harmbench_behaviors_text_all.csv` (HF mirror `harmbench/harmbench` tried first) | MIT, © 2024 Center for AI Safety |
| JailbreakBench (optional) | `JailbreakBench/JBB-Behaviors` (HF) | MIT |

Both AdvBench and HarmBench are **MIT-licensed**, so redistribution *with the copyright
notice* is technically permitted — we nonetheless fetch-at-setup rather than commit, for
responsible-use reasons (these are red-teaming corpora) and to keep this repo publishable.
If you cite the data, cite the original papers (AdvBench: Zou et al. 2023; HarmBench:
Mazeika et al. 2024), not this repo.

## What lives under `data/` locally (git-ignored)

```
data/
├── README.md                  # this file (the only tracked thing under data/)
├── raw/                        # advbench.csv, harmbench.csv, manifest.json  (downloaded)
├── processed/                  # prompts.csv  (876 AdvBench+HarmBench prompts; harmful)
└── results/
    ├── labels/                 # per-run CSVs: prompt_id,prompt,response,label,…  (harmful: raw completions)
    └── metrics/                # derived aggregate stats (numbers only; no prompt/response text)
```

`data/results/labels/*.csv` contain raw model completions to harmful prompts and must not
be published. To reproduce the analysis, run the pipeline (`scripts/01…07`) locally.
