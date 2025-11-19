"""
Comparison tool: aggregates metrics JSON files into a CSV table. (PPO and QRL comparison)
"""

import os
import glob
import json
import pandas as pd
from typing import List, Dict, Any


def load_metrics_from_folder(folder: str, pattern: str = "*.json") -> pd.DataFrame:
    folder = folder.rstrip("/")

    files = glob.glob(os.path.join(folder, pattern), recursive=True)
    rows = []

    for fp in files:
        try:
            with open(fp, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        flat: Dict[str, Any] = {}

        def flatten(obj: Dict[str, Any], prefix=""):
            for k, v in obj.items():
                key = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
                if isinstance(v, dict):
                    flatten(v, key)
                else:
                    flat[key] = v

        if isinstance(data, dict):
            flatten(data)
            flat["_source"] = fp
            rows.append(flat)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def compare_and_save(folder: str, out_csv: str):
    """
    Legacy mode: scan all JSON metrics in a folder and produce a flat CSV.
    Useful for quickly browsing many validation files from PPO/QRL runs.
    """
    df = load_metrics_from_folder(folder)
    if df.empty:
        print("[compare] No metrics found in folder:", folder)
        return None

    # Identify key metric columns
    metric_suffixes = [
        "cumulative_return",
        "annualized_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "cvar95",
    ]

    keep_cols = ["_source"]
    for suf in metric_suffixes:
        matches = [c for c in df.columns if c.endswith(suf)]
        keep_cols.extend(matches)

    keep_cols = list(dict.fromkeys(keep_cols))  # unique

    small = df[keep_cols].copy()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    small.to_csv(out_csv, index=False)

    # Save summary statistics
    summary_path = out_csv.replace(".csv", "_summary.csv")
    small.describe(include="all").to_csv(summary_path)

    print("[compare] Saved:", out_csv)
    print("[compare] Summary:", summary_path)
    return small


def _load_flat(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    # final metrics are already flat; just in case, keep them as-is
    return data


def compare_single_runs(ppo_final_path: str, qrl_final_path: str, out_csv: str = None) -> pd.DataFrame:
    """
    Load one PPO final metrics JSON and one QRL final metrics JSON,
    and produce a small table comparing core metrics.

    Example:
        python -m src.utils.comparison \
            --ppo_final results/ppo_.../final_metrics.json \
            --qrl_final results/qrl_.../final.json \
            --out results/ppo_qrl_comparison.csv
    """
    if not os.path.exists(ppo_final_path):
        raise FileNotFoundError(f"PPO final metrics not found: {ppo_final_path}")
    if not os.path.exists(qrl_final_path):
        raise FileNotFoundError(f"QRL final metrics not found: {qrl_final_path}")

    ppo = _load_flat(ppo_final_path)
    qrl = _load_flat(qrl_final_path)

    ppo_row = {"algo": "ppo"}
    ppo_row.update({k: v for k, v in ppo.items() if k not in ["pv", "returns"]})

    qrl_row = {"algo": "qrl"}
    qrl_row.update({k: v for k, v in qrl.items() if k not in ["pv", "returns"]})

    df = pd.DataFrame([ppo_row, qrl_row])

    if out_csv is not None:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print("[compare] Pairwise PPO vs QRL table saved:", out_csv)

    # Print a human-readable summary
    p_sharpe = float(ppo.get("sharpe", 0.0))
    q_sharpe = float(qrl.get("sharpe", 0.0))
    p_cum = float(ppo.get("cumulative_return", 0.0))
    q_cum = float(qrl.get("cumulative_return", 0.0))

    print("\n=== PPO vs QRL FINAL METRICS ===")
    print(f"PPO: Sharpe={p_sharpe:.3f}, CumRet={p_cum:.3f}")
    print(f"QRL: Sharpe={q_sharpe:.3f}, CumRet={q_cum:.3f}")

    if q_sharpe > p_sharpe:
        print("[RESULT] Quantum RL (QRL) outperformed PPO on Sharpe in THIS run.")
    elif q_sharpe < p_sharpe:
        print("[RESULT] PPO outperformed QRL on Sharpe in THIS run.")
    else:
        print("[RESULT] PPO and QRL have equal Sharpe in THIS run.")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_folder", type=str, default=None,
                        help="Folder of JSON metrics to aggregate (legacy mode).")
    parser.add_argument("--out", type=str, default="results/comparison_table.csv")
    parser.add_argument("--ppo_final", type=str, default=None,
                        help="Path to PPO final_metrics.json")
    parser.add_argument("--qrl_final", type=str, default=None,
                        help="Path to QRL final.json")

    args = parser.parse_args()

    if args.ppo_final and args.qrl_final:
        # Direct PPO vs QRL comparison
        compare_single_runs(args.ppo_final, args.qrl_final, args.out)
    elif args.results_folder:
        # Legacy folder aggregation
        compare_and_save(args.results_folder, args.out)
    else:
        print("Usage:")
        print("  Pairwise comparison:")
        print("    python -m src.utils.comparison --ppo_final <ppo_final.json> --qrl_final <qrl_final.json> --out <csv>")
        print("  Folder aggregation:")
        print("    python -m src.utils.comparison --results_folder <folder> --out <csv>")
