#!/usr/bin/env bash
set -euo pipefail

for config in configs/lastfm_phase_a.yaml configs/foursquare_phase_a.yaml configs/ml1m_phase_a.yaml; do
  for seed in 2022 2023 2024 2025 2026; do
    python train_phase_a.py --config "$config" --seed "$seed"
  done
done
