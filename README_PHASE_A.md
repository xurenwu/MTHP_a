# MTHP-HC Phase A: clean paper reproduction

This directory set is a complete, unified reimplementation of the paper model rather than a patch of the legacy `HTSER_a` code. It keeps the supplied `.lst`, `in_degree.npy`, and `item_matrix*.npy` files unchanged and uses them through a memory-safe graph interface.

## What is implemented

1. Separate user and item ID embeddings.
2. User-item hypergraph-induced item graph supplied by `item_matrix*.npy`.
3. User-specific matrix slicing for every history sequence.
4. Symmetric degree normalization using the supplied `in_degree.npy` when available.
5. Four-layer SHCN by default, based on the layer count in the legacy code.
6. Structural diffusion with the paper relation term `A_hat + delta * Norm(QK^T/sqrt(d))`.
7. Mean pooling and normalized fusion with the personalized user embedding.
8. Hour/day/week Hawkes branches.
9. Learnable global multi-granularity weights.
10. User-specific temporal decay.
11. Paper-faithful cosine base/excitation scores and BPR pairwise loss.
12. Full-item ranking with Recall, MRR, and NDCG.
13. Normal next-item and user-unseen next-item evaluation.
14. Validation-based early stopping, checkpoints, deterministic seeds, and five-seed execution.

## Configuration provenance

| Setting | Value | Source |
|---|---:|---|
| embedding dimension | 256 | paper and legacy config |
| negative samples | 5 | paper and legacy config |
| history length | 3 | paper and legacy config |
| learning rate | 1e-3 | paper and legacy config |
| epochs | 200 | paper and legacy config |
| batch size | 512 | paper |
| weight decay | 0.01 | paper |
| SHCN layers | 4 | legacy `Model_new_user_attention3.py` |
| SHCN heads | 1 | legacy code |
| graph relation coefficient | 0.1 | legacy `hGCN.py` |
| SHCN dropout | 0.1 | legacy layer definition |

Paper-unspecified settings are explicit YAML options and are never hidden in source code.

## Dataset paths

The supplied layout is expected:

```text
data_mthp_hgcn/
  data_lastfm/
  data_foursquare/
  data_ml_1m/
```

Update only the file paths in `configs/*.yaml` when the matrix or degree filenames differ. Dense `.npy` matrices are memory-mapped and are not fully loaded into RAM. Sparse `.npz` matrices are also supported.

When a graph file is absent and `build_if_missing: true`, a training-only sparse item graph is generated as a cache; the original `.lst` files are not modified.

## Run

```bash
pip install -r requirements-phase-a.txt
python train_phase_a.py --config configs/lastfm_phase_a.yaml --seed 2026
```

Run all three datasets and five seeds:

```bash
bash scripts/run_phase_a.sh
```

## Reproduction boundary

Phase A reproduces the method stated in the paper. It intentionally does **not** add the Phase-B corrections such as Softplus non-negative intensities, user-granularity-specific decay, reliability-aware graph edges, or a temporal point-process likelihood. `positive_intensity` is disabled in all Phase-A configurations.

The legacy repository used a signed ranking score while calling it a Hawkes intensity. Phase A preserves that behavior for controlled reproduction, but the code names and documentation distinguish the score from a mathematically valid non-negative point-process intensity.

## Important protocol choices

- The provided train and test interactions remain unchanged.
- One final interaction per user is taken from the provided training file for validation. Set `validation_items_per_user: 0` to disable this.
- Test histories are seeded with the complete provided training history, then updated chronologically with preceding test events.
- Full ranking is used.
- Seen items are not masked by default, matching repeat-aware music recommendation. Set `mask_seen_items: true` for conventional non-repeat evaluation.
- The user-unseen task means an item not previously consumed by that user; it is not global cold start.

## Data audit and result aggregation

Before training, verify IDs, matrix dimensions, and degree dimensions:

```bash
python scripts/check_data.py --config configs/lastfm_phase_a.yaml
```

After five seeds:

```bash
python scripts/aggregate_results.py outputs/phase_a/lastfm_phase_a
```

For machines that cannot handle the 30,000-item experiment immediately, `configs/lastfm_top10000_phase_a.yaml` uses the supplied top-10,000 split and matrix. This is a debugging configuration, not a replacement for the paper-scale LastFM result.
