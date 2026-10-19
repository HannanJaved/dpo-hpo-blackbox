# DPO-HPO Blackbox

Offline **multi-fidelity blackbox** for DPO hyperparameter selection (learning rate and β): checkpoint evaluation tables, construction code, and HPO simulation traces.

Use this repository to:

- load a ready-made `(lr, β, fidelity) → benchmark scores` grid
- compare HPO methods on released search traces
- rebuild blackbox tables or (optionally) re-run [Syne Tune](https://github.com/syne-tune/syne-tune) simulations

**Repo size:** ~425 MB (mostly gzipped traces). Clone with Git; no LFS required.

## Quickstart

```bash
pip install pandas
```

```python
import pandas as pd

bb = pd.read_csv("data/all_checkpoints_blackbox.csv")
# One model size
qwen = bb[(bb.family == "qwen3") & (bb["size"] == "0.6b")]
# Scores at the final fidelity for each (lr, beta)
final = qwen.groupby(["lr", "beta"], as_index=False).last()
print(final[["lr", "beta", "Z-Macro", "ELO"]].sort_values("Z-Macro", ascending=False).head())
```

Load a released HPO trace (no full re-simulation needed):

```python
import pandas as pd

inc = pd.read_csv(
    "traces/qwen3_0.6b/gp/generous/best_found.csv.gz",
    compression="gzip",
)
print(inc.groupby(["search_objective", "optimizer"])["Z-Macro"].mean())
```

Smoke-check that tables match the manifest:

```bash
python code/quickstart_check.py
```

## What’s included

| Path | Contents |
|---|---|
| `data/` | Checkpoint grids (core artifact): raw + blackbox CSVs |
| `traces/` | Qwen3 informed-protocol HPO search logs (gzipped) |
| `code/` | Blackbox construction, surrogates, [Syne Tune](https://github.com/syne-tune/syne-tune) runners |
| `protocol.json` | HPO settings (ASHA rungs, budgets, seeds, …) |
| `manifest.json` | Per-size config / fidelity counts |
| `traces_manifest.json` | Per-trace paths, row counts, objectives |

**Qwen3** (tables + traces): 0.6B, 1.7B, 4B, 8B, 14B  
**Llama** (tables only): 1B, 3B, 8B

### Checkpoint schedules

- **[Dolci-Instruct-DPO](https://huggingface.co/datasets/allenai/Dolci-Instruct-DPO) (default blackbox):** each `(lr, β)` is scored at DPO steps 200, 400, …, 2000 and the final checkpoint (~2031). Cost is `elapsed_time_sec` from that run’s trainer runtime.
- **[UltraFeedback](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences)** (`qwen3_*_ultrafeedback_*.csv`): shorter schedule (steps 48, 96, …, 432 and final ~478). These files include `dataset=ultrafeedback_binarized` and are **not** mixed into `all_checkpoints_*.csv`.

## Layout detail

```
data/
  {qwen3,llama}_{size}_raw.csv
  {qwen3,llama}_{size}_blackbox.csv
  all_checkpoints_raw.csv              Qwen3 + Llama Dolci-Instruct-DPO
  all_checkpoints_blackbox.csv
  llama_all_checkpoints_{raw,blackbox}.csv
  qwen3_{size}_ultrafeedback_{raw,blackbox}.csv
traces/
  qwen3_{size}/{gp,knn1,knn3,knn5}/{generous,tight}/
    simulation_raw.csv.gz     every optimizer report (config × checkpoint)
    best_found.csv.gz         one incumbent per (objective, optimizer, seed)
code/                         construction + simulation
```

## Core table schema

`all_checkpoints_*.csv` is indexed by `(family, size, lr, beta, dpo_step)`.

| column | meaning |
|---|---|
| `family`, `size` | model family and parameter count |
| `lr`, `beta` | DPO learning rate and β |
| `dpo_step` | fidelity (training step) |
| `elapsed_time_sec` | cumulative training cost to that checkpoint |
| `ARC-C` … `ELO` | 11 downstream scores (higher is better) |
| `Z-Static`, `Z-Dynamic`, `Z-All`, `Z-Macro` | **blackbox only** — per-fidelity z-scores (static / preference / 11-bench mean / ½(dynamic+static)) |

`*_raw.csv` keeps empty cells where an evaluation was missing. `*_blackbox.csv` applies the construction pipeline: drop `(lr, β)` pairs without a full fidelity trajectory; forward/back-fill rare intra-config gaps; z-normalize each benchmark **within a size and fidelity**, then average. Rebuild with:

```bash
pip install -r requirements.txt   # or: pip install pandas numpy
python code/rebuild_blackbox_tables.py
```

## Simulation traces

One row per blackbox query. Optimizers: RandomSearch, TPE, CQR, ASHA, BOHB. Search objectives: Z-Dynamic, Z-All, Z-Macro. 25 seeds. Surrogates: GP (primary) and kNN-1/3/5. Budgets: generous = 8× one full DPO run, tight = 1×. ASHA/BOHB grace period 400, reduction factor 2 (rungs 400 / 800 / 1600). See `protocol.json`.

| column | meaning |
|---|---|
| `search_objective`, `optimizer`, `seed`, `trial_id` | which run |
| `dpo_step`, `st_tuner_time` | fidelity and simulated wall-clock |
| `lr`, `beta` | proposed hyperparameters |
| benchmark / Z-* columns | surrogate-evaluated scores at that checkpoint |

`best_found.csv.gz` stores the incumbent under the finished-trial rule in `code/select_incumbent.py`. Re-running simulators is optional and heavier; see [`code/README.md`](code/README.md).

## FAQ

**Raw vs blackbox?**  
Raw keeps missing evals as empty. Blackbox drops incomplete `(lr, β)` trajectories, fills rare gaps, and adds Z-* composites for HPO objectives.

**Dolci-Instruct-DPO vs UltraFeedback?**  
Default tables use [Dolci-Instruct-DPO](https://huggingface.co/datasets/allenai/Dolci-Instruct-DPO). [UltraFeedback](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences) files are a separate preference dataset and shorter training schedule; use them only when you intend that setting.

**Why no Llama traces?**  
Released HPO traces cover the Qwen3 informed continuous-surrogate study. Llama appears in the checkpoint tables only (`protocol.json`: `llama_in_tables_only`).

**Do I need [Syne Tune](https://github.com/syne-tune/syne-tune) to use the dataset?**  
No. Tables and traces are plain CSV/CSV.GZ. Syne Tune is only needed to rebuild serialized blackboxes or re-run simulations.

## Citation

If you use this dataset, please cite (BibTeX also in [`CITATION.cff`](CITATION.cff)). An arXiv ID will be added once the paper is online:

```bibtex
@misc{dpo-hpo-blackbox,
  title        = {DPO-HPO Blackbox: Offline Multi-Fidelity Benchmarks for DPO Hyperparameter Selection},
  author       = {Javed, Hannan and Salinas, David and Golebiowski, Jacek and Klein, Aaron},
  year         = {2026},
  howpublished = {\url{https://github.com/HannanJaved/dpo-hpo-blackbox}}
}
```

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see [`LICENSE`](LICENSE).

## Contact

Issues and questions: [github.com/HannanJaved/dpo-hpo-blackbox/issues](https://github.com/HannanJaved/dpo-hpo-blackbox/issues).
