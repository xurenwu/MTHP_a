#!/usr/bin/env bash
set -euo pipefail

for config in configs/asoc/lastfm_10k.yaml; do
  for seed in 2022 2023 2024 2025 2026; do
    python train_phase_a.py --config "$config" --seed "$seed"
  done
done
