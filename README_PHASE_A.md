# MTHP-HC reproduction and ASOC extension

This repository contains two deliberately separated experiment paths:

1. **Reproduction path**: follows the equations and stated hyperparameters of the paper as closely as the supplied files allow.
2. **ASOC path**: uses leakage-safe validation graphs and mathematically valid positive temporal intensities for the journal extension.

The repository data split does not exactly match every statistic reported in the paper. Results obtained from the supplied files must therefore be described as repository-split results rather than an exact reproduction of the published tables.

## Fixed implementation issues

The current implementation fixes the following problems:

- explicit `train`, `valid`, `test`, and `test_new` split support;
- test histories are seeded with all preceding train/validation events;
- the supplied new-item split is no longer ignored;
- ambiguous automatic graph-file selection now raises an error;
- validation leakage is blocked when `strict_no_leakage: true`;
- training-only sparse graphs can be rebuilt deterministically;
- graph degree, item popularity, and local subgraph degree are no longer conflated;
- self-loops use `auto`, `add`, or `none`, avoiding accidental double self-loops;
- SHCN follows `V=ELU(XW)` before structural propagation;
- the undocumented extra SHCN output projection is disabled by default;
- structural-only user representations fall back to the user embedding for empty histories;
- positive decay can use a differentiable Softplus parameterization;
- user-granularity decay and user-adaptive granularity weights are supported;
- reproduction runs can save the final epoch while validation runs save the best epoch;
- `full_ranking: false` no longer silently behaves as full ranking;
- data auditing reports matrix diagonals and verifies supplied degree vectors.

## Dataset mapping

The supplied Last.fm-derived files are mapped as follows:

- `top10000` -> repository LastFM-10K split;
- `top30000` -> repository 30Music-30K split;
- `foursquare_10000` -> repository Foursquare legacy split.

Before citing paper-scale dataset statistics, run the data audit and compare the actual counts with the manuscript.

## Reproduction configuration

The main reproduction files are:

```text
configs/lastfm_phase_a.yaml
configs/30music_phase_a.yaml
configs/foursquare_phase_a.yaml
```

They use:

- history length 3;
- cosine base and excitation scores;
- shared user decay;
- global learnable hour/day/week weights;
- signed ranking scores (`positive_intensity: false`);
- one SHCN layer following the restored paper equation;
- no internal validation holdout;
- final-epoch checkpointing;
- full-item ranking;
- no seen-item masking.

Run:

```bash
pip install -r requirements-phase-a.txt
python scripts/check_data.py --config configs/lastfm_phase_a.yaml --compare-degree
python train_phase_a.py --config configs/lastfm_phase_a.yaml --seed 2026
```

Run three repository datasets with five seeds:

```bash
bash scripts/run_phase_a.sh
```

## ASOC configuration

`configs/asoc/lastfm_10k.yaml` demonstrates the leakage-safe journal path:

- an internal validation split is created;
- the item graph is rebuilt from training-only interactions;
- positive Softplus intensity is enabled;
- decay is user- and granularity-specific;
- temporal granularity weights are user-adaptive;
- residual SHCN propagation and early stopping are enabled.

These choices are an extension of the paper model and must not be reported as the original reproduction result.

## Graph handling

Dense `.npy` matrices are memory-mapped. Sparse `.npz` matrices are supported directly. Only the local `L x L` history graph is transferred to the accelerator.

Graph options:

```yaml
graph:
  self_loop_mode: auto   # auto, add, none
  degree_source: local   # local, file, matrix
```

Use `degree_source: file` only with a verified item-item graph degree vector. Do not use an interaction-popularity vector as graph degree.

For a clean validation protocol:

```yaml
graph:
  rebuild_from_train: true
  strict_no_leakage: true
```

## Evaluation tasks

- Normal next-item evaluation uses the configured `test_file`.
- User-unseen next-item evaluation uses `test_new_file` when provided.
- User-unseen means unseen by that user before prediction; it is not global item cold start.
- Full ranking is the only implemented evaluation mode.

## Tests

```bash
pytest -q
```

The tests cover ID alignment, explicit split handling, test-history seeding, graph self-loops, SHCN gradients, positive decay, adaptive granularity weights, and empty-history fallback.
