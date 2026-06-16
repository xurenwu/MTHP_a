#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(value, (int, float)):
        out[prefix] = float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="e.g. outputs/phase_a/lastfm_phase_a")
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.run_dir).glob("seed_*/results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {"seed": path.parent.name}
        flatten("", {"test": data.get("test"), "new_item_test": data.get("new_item_test")}, row)
        rows.append(row)
    if not rows:
        raise SystemExit("No seed_*/results.json files found")
    keys = sorted(k for k in rows[0] if k != "seed")
    summary = {"runs": len(rows)}
    for key in keys:
        values = np.asarray([r[key] for r in rows if key in r], dtype=float)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    output = Path(args.run_dir) / "summary_mean_std.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
