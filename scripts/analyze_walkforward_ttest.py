"""
T-test analysis on all 15 fold values (5 folds x 3 seeds) per model.

Uses per-fold balanced_accuracy instead of only 3 mean-per-seed aggregates.
Paired comparisons align observations by (seed, fold).
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEFAULT_RESULTS_FILE = os.path.join(MODELS_DIR, "btc_walkforward_results_detailed.json")
DEFAULT_OUTPUT_FILE = os.path.join(MODELS_DIR, "btc_walkforward_ttest_results.json")
BASELINE = 0.5
ALPHA = 0.05
METRIC = "balanced_accuracy"


def load_results(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_fold_series(results: list, model: str, metric: str = METRIC) -> List[Dict]:
    """Return one record per fold with explicit seed/fold keys."""
    series = []
    for entry in sorted(
        [r for r in results if r["model"] == model],
        key=lambda r: (r["seed"], r.get("n_folds", 0)),
    ):
        for fold_idx, fold in enumerate(entry["fold_metrics"], start=1):
            series.append(
                {
                    "model": model,
                    "seed": entry["seed"],
                    "fold": fold_idx,
                    "value": float(fold[metric]),
                }
            )
    return series


def series_to_array(series: List[Dict]) -> np.ndarray:
    return np.array([row["value"] for row in series], dtype=float)


def align_paired(series_a: List[Dict], series_b: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Pair observations by (seed, fold)."""
    index_b = {(row["seed"], row["fold"]): row["value"] for row in series_b}
    paired_a, paired_b = [], []
    for row in series_a:
        key = (row["seed"], row["fold"])
        if key in index_b:
            paired_a.append(row["value"])
            paired_b.append(index_b[key])
    return np.array(paired_a, dtype=float), np.array(paired_b, dtype=float)


def mean_per_seed_values(results: list, model: str, metric_key: str = "mean_balanced_accuracy") -> np.ndarray:
    return np.array(
        [float(r[metric_key]) for r in results if r["model"] == model],
        dtype=float,
    )


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    if diff.std(ddof=1) == 0:
        return 0.0
    return float(diff.mean() / diff.std(ddof=1))


def cohens_d_one_sample(x: np.ndarray, mu: float = BASELINE) -> float:
    if x.std(ddof=1) == 0:
        return 0.0
    return float((x.mean() - mu) / x.std(ddof=1))


def summarize_sample(name: str, values: np.ndarray) -> Dict:
    return {
        "name": name,
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
        "values": values.tolist(),
    }


def run_ttest_report(
    results: list,
    models: List[str],
    metric: str = METRIC,
    baseline: float = BASELINE,
    alpha: float = ALPHA,
) -> Dict:
    report = {
        "metric": metric,
        "baseline": baseline,
        "alpha": alpha,
        "samples": {},
        "paired_tests": [],
        "one_sample_tests": [],
        "incorrect_mean_per_seed_tests": [],
    }

    model_series = {model: extract_fold_series(results, model, metric) for model in models}
    for model, series in model_series.items():
        values = series_to_array(series)
        report["samples"][model] = summarize_sample(model, values)

    # One-sample t-tests on all 15 fold values
    for model in models:
        values = series_to_array(model_series[model])
        if len(values) < 2:
            continue
        t_stat, p_value = stats.ttest_1samp(values, baseline)
        report["one_sample_tests"].append(
            {
                "model": model,
                "comparison": f"{model} vs baseline ({baseline})",
                "n": int(len(values)),
                "mean": float(values.mean()),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < alpha),
                "cohens_d": cohens_d_one_sample(values, baseline),
            }
        )

    # Paired model-vs-model tests on matched (seed, fold)
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            paired_a, paired_b = align_paired(model_series[model_a], model_series[model_b])
            if len(paired_a) < 2:
                continue
            t_stat, p_value = stats.ttest_rel(paired_a, paired_b)
            report["paired_tests"].append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "comparison": f"{model_a} vs {model_b}",
                    "n_pairs": int(len(paired_a)),
                    "mean_a": float(paired_a.mean()),
                    "mean_b": float(paired_b.mean()),
                    "mean_diff": float((paired_a - paired_b).mean()),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": bool(p_value < alpha),
                    "cohens_d_paired": cohens_d_paired(paired_a, paired_b),
                }
            )

    # Incorrect approach for reference: only 3 mean-per-seed values
    for model in models:
        seed_means = mean_per_seed_values(results, model)
        if len(seed_means) < 2:
            continue
        t_stat, p_value = stats.ttest_1samp(seed_means, baseline)
        report["incorrect_mean_per_seed_tests"].append(
            {
                "model": model,
                "comparison": f"{model} vs baseline ({baseline}) [WRONG: mean-per-seed only]",
                "n": int(len(seed_means)),
                "mean": float(seed_means.mean()),
                "values": seed_means.tolist(),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < alpha),
                "note": "Underpowered: uses only one aggregate per seed instead of all fold values.",
            }
        )

    return report


def print_report(report: Dict) -> None:
    print("=" * 80)
    print("T-TEST ANALYSIS ON ALL FOLD VALUES (5 folds x 3 seeds = 15 per model)")
    print("=" * 80)

    for model, sample in report["samples"].items():
        print(f"\n{model}: n={sample['n']}")
        print(f"  Mean: {sample['mean']:.4f}")
        print(f"  Std:  {sample['std']:.4f}")
        print(f"  Min:  {sample['min']:.4f}")
        print(f"  Max:  {sample['max']:.4f}")

    print("\n" + "=" * 80)
    print("PAIRED T-TESTS (matched by seed + fold)")
    print("=" * 80)
    for test in report["paired_tests"]:
        print(f"\n{test['comparison']}")
        print(f"  n_pairs: {test['n_pairs']}")
        print(f"  mean_a: {test['mean_a']:.4f}")
        print(f"  mean_b: {test['mean_b']:.4f}")
        print(f"  mean_diff: {test['mean_diff']:.4f}")
        print(f"  t-statistic: {test['t_statistic']:.4f}")
        print(f"  p-value: {test['p_value']:.4f}")
        print(f"  significant (alpha={report['alpha']}): {'YES' if test['significant'] else 'NO'}")
        print(f"  Cohen's d (paired): {test['cohens_d_paired']:.4f}")

    print("\n" + "=" * 80)
    print("ONE-SAMPLE T-TESTS VS BASELINE (all fold values)")
    print("=" * 80)
    for test in report["one_sample_tests"]:
        print(f"\n{test['comparison']}")
        print(f"  n: {test['n']}")
        print(f"  mean: {test['mean']:.4f}")
        print(f"  t-statistic: {test['t_statistic']:.4f}")
        print(f"  p-value: {test['p_value']:.4f}")
        print(f"  significant (alpha={report['alpha']}): {'YES' if test['significant'] else 'NO'}")
        print(f"  Cohen's d: {test['cohens_d']:.4f}")

    print("\n" + "=" * 80)
    print("INCORRECT APPROACH (for comparison): mean-per-seed only (n=3)")
    print("=" * 80)
    for test in report["incorrect_mean_per_seed_tests"]:
        print(f"\n{test['comparison']}")
        print(f"  n: {test['n']}")
        print(f"  values: {[round(v, 4) for v in test['values']]}")
        print(f"  mean: {test['mean']:.4f}")
        print(f"  p-value: {test['p_value']:.4f}")
        print(f"  significant: {'YES' if test['significant'] else 'NO'}")
        print(f"  note: {test['note']}")


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward t-tests on all fold values.")
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS_FILE,
        help="Path to btc_walkforward_results_detailed.json",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Path to save JSON report",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to compare (default: all models found in results)",
    )
    args = parser.parse_args()

    results = load_results(args.results)
    models = args.models or sorted({r["model"] for r in results})
    report = run_ttest_report(results, models=models)
    print_report(report)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
