# Construction code

Python used to clean the checkpoint grid, fit continuous surrogates, and run
five Syne Tune optimizers. Released **tables** and **traces** under `../data`
and `../traces` are enough for most users; this directory is for reproduction
and extension.

## Setup

```bash
# From the repository root
pip install -r requirements.txt
```

- **Tables / quickstart only:** `pandas` (+ `numpy` to rebuild blackbox CSVs).
- **Surrogates + HPO sims:** also `syne-tune`, `scikit-learn`, `matplotlib`.
  Simulations write under `../work/` (gitignored).

## Scripts

| file | role |
|---|---|
| `quickstart_check.py` | verify table row counts vs `manifest.json`; peek at one trace |
| `build_blackbox.py` | drop incomplete `(lr, β)` trajectories, fill gaps, z-score composites (incl. Z-Macro); serialize Syne Tune blackbox |
| `rebuild_blackbox_tables.py` | pandas-only rebuild of `data/*_blackbox.csv` from `data/*_raw.csv` |
| `build_blackbox_surrogate.py` | GP / kNN surrogate over log10(lr) × log10(β) |
| `run_simulations_surrogate.py` | RandomSearch, TPE, CQR, ASHA, BOHB × 25 seeds |
| `schedulers.py` | ASHA/BOHB rungs and seeded RandomSearch/TPE |
| `select_incumbent.py` | finished-trial incumbent (not argmax over every checkpoint) |
| `analyze_results.py` / `analyze_results_surrogate.py` | regret-style plots from simulation outputs |
| `common.py` | protocol constants; paths resolve to this repo |

## Common commands

```bash
# Sanity-check the released artifact
python code/quickstart_check.py

# Rebuild blackbox CSVs from raw grids
python code/rebuild_blackbox_tables.py

# Fit a surrogate and run HPO (writes under work/)
python code/build_blackbox_surrogate.py --size 0.6b --family qwen3 --surrogate gp
python code/run_simulations_surrogate.py --size 0.6b --family qwen3 --surrogate gp --budget-tag generous
```

Released traces under `../traces/` are already the completed searches; re-running
the simulators is optional and can take substantial CPU time.
