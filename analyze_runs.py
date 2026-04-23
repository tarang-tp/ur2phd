from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze order-sensitivity results across SST-2 runs.")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--top-disagreement-quantile", type=float, default=0.9)
    return parser.parse_args()


def load_validation_predictions(artifacts_root: Path) -> pd.DataFrame:
    frames = []
    for prediction_file in sorted(artifacts_root.glob("*/validation_predictions.csv")):
        frames.append(pd.read_csv(prediction_file))
    if not frames:
        raise FileNotFoundError(f"No validation prediction files found under {artifacts_root}.")
    return pd.concat(frames, ignore_index=True)


def load_run_metrics(artifacts_root: Path) -> pd.DataFrame:
    rows = []
    for metrics_file in sorted(artifacts_root.glob("*/metrics.json")):
        payload = json.loads(metrics_file.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_name": payload["run_name"],
                "permutation_seed": payload["permutation_seed"],
                "model_seed": payload["model_seed"],
                "eval_accuracy": payload["metrics"]["eval_accuracy"],
                "eval_loss": payload["metrics"]["eval_loss"],
            }
        )
    if not rows:
        raise FileNotFoundError(f"No metrics files found under {artifacts_root}.")
    return pd.DataFrame(rows).sort_values("run_name").reset_index(drop=True)


def build_sentence_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    from ur2phd.experiment import add_text_features

    enriched = add_text_features(
        predictions[["example_id", "sentence", "label", "run_name", "prediction", "prob_positive", "confidence"]]
    )
    grouped = enriched.groupby("example_id", as_index=False)
    summary = grouped.agg(
        sentence=("sentence", "first"),
        label=("label", "first"),
        run_count=("run_name", "nunique"),
        majority_vote_count=("prediction", lambda s: int(s.value_counts().max())),
        disagreement_count=("prediction", lambda s: int(len(s) - s.value_counts().max())),
        majority_label=("prediction", lambda s: int(s.value_counts().idxmax())),
        mean_confidence=("confidence", "mean"),
        confidence_std=("prob_positive", "std"),
        token_length=("token_length", "first"),
        has_negation=("has_negation", "first"),
        has_contrast=("has_contrast", "first"),
    )
    summary["is_unanimous"] = summary["disagreement_count"] == 0
    return summary


def build_pairwise_agreement(predictions: pd.DataFrame) -> pd.DataFrame:
    pivot = predictions.pivot(index="example_id", columns="run_name", values="prediction").sort_index(axis=1)
    run_names = pivot.columns.tolist()
    agreement = pd.DataFrame(index=run_names, columns=run_names, dtype=float)
    for left in run_names:
        for right in run_names:
            agreement.loc[left, right] = float((pivot[left] == pivot[right]).mean())
    return agreement


def build_model_scores(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = predictions.pivot(index="example_id", columns="run_name", values="prediction").sort_index(axis=1)
    labels = predictions.groupby("example_id")["label"].first().loc[pivot.index]

    rows = []
    for run_name in pivot.columns:
        other_columns = [column for column in pivot.columns if column != run_name]
        other_votes = pivot[other_columns]
        other_majority = other_votes.mode(axis=1)
        majority_label = other_majority[0].astype(int)
        agreement_score = float((pivot[run_name] == majority_label).mean())
        tied_rows = other_majority.shape[1] > 1
        if (~tied_rows).any():
            strict_agreement_score = float((pivot.loc[~tied_rows, run_name] == majority_label.loc[~tied_rows]).mean())
        else:
            strict_agreement_score = float("nan")

        rows.append(
            {
                "run_name": run_name,
                "agreement_score": agreement_score,
                "agreement_score_excluding_ties": strict_agreement_score,
                "validation_accuracy_from_predictions": float((pivot[run_name] == labels).mean()),
                "tie_rate": float(tied_rows.mean()),
            }
        )

    scores = pd.DataFrame(rows).merge(metrics, on="run_name", how="left")
    return scores.sort_values("agreement_score", ascending=False).reset_index(drop=True)


def summarize_groups(summary: pd.DataFrame, quantile: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = summary["disagreement_count"].quantile(quantile)
    unstable = summary[summary["disagreement_count"] >= threshold].copy()
    stable = summary[summary["is_unanimous"]].copy()
    return stable, unstable


def compare_groups(stable: pd.DataFrame, unstable: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import ttest_ind

    comparisons = []
    if stable.empty or unstable.empty:
        return pd.DataFrame(
            [
                {
                    "feature": "group_comparison_unavailable",
                    "stable_mean": np.nan,
                    "unstable_mean": np.nan,
                    "test": "not_run",
                    "statistic": np.nan,
                    "p_value": np.nan,
                }
            ]
        )

    numeric_columns = ["token_length", "mean_confidence", "confidence_std"]
    for column in numeric_columns:
        statistic, p_value = ttest_ind(
            stable[column],
            unstable[column],
            equal_var=False,
            nan_policy="omit",
        )
        comparisons.append(
            {
                "feature": column,
                "stable_mean": float(stable[column].mean()),
                "unstable_mean": float(unstable[column].mean()),
                "test": "welch_ttest",
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        )

    for column in ["has_negation", "has_contrast"]:
        comparisons.append(
            {
                "feature": column,
                "stable_mean": float(stable[column].mean()),
                "unstable_mean": float(unstable[column].mean()),
                "test": "difference_in_proportions",
                "statistic": float(unstable[column].mean() - stable[column].mean()),
                "p_value": np.nan,
            }
        )

    return pd.DataFrame(comparisons)


def create_plots(summary: pd.DataFrame, agreement: pd.DataFrame, scores: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(summary["disagreement_count"], bins=range(0, int(summary["disagreement_count"].max()) + 2), discrete=True)
    plt.title("Per-Sentence Disagreement Counts")
    plt.xlabel("Minority-vote count")
    plt.ylabel("Number of validation sentences")
    plt.tight_layout()
    plt.savefig(output_dir / "disagreement_histogram.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(agreement, annot=True, fmt=".3f", cmap="viridis", square=True, cbar_kws={"label": "Agreement"})
    plt.title("Pairwise Agreement Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "pairwise_agreement_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(summary["confidence_std"].fillna(0), bins=30)
    plt.title("Confidence Variance Distribution")
    plt.xlabel("Std. dev. of positive-class probability")
    plt.ylabel("Number of validation sentences")
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_variance_histogram.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=scores, x="agreement_score", y="eval_accuracy", s=90)
    for row in scores.itertuples():
        plt.text(row.agreement_score + 0.0005, row.eval_accuracy + 0.0005, row.run_name, fontsize=8)
    plt.title("Agreement Score vs Validation Accuracy")
    plt.xlabel("Agreement score")
    plt.ylabel("Validation accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "agreement_vs_accuracy.png", dpi=200)
    plt.close()


def build_summary_report(
    summary: pd.DataFrame,
    agreement: pd.DataFrame,
    metrics: pd.DataFrame,
    scores: pd.DataFrame,
    group_comparison: pd.DataFrame,
) -> dict:
    from scipy.stats import pearsonr

    upper_triangle = agreement.where(~np.tril(np.ones(agreement.shape, dtype=bool)))
    if len(scores) >= 2:
        pearson_r, pearson_p = pearsonr(scores["agreement_score"], scores["eval_accuracy"])
    else:
        pearson_r, pearson_p = float("nan"), float("nan")
    best_by_agreement = scores.iloc[0]
    top_3_by_accuracy = metrics.sort_values("eval_accuracy", ascending=False).head(3)["run_name"].tolist()

    return {
        "num_runs": int(metrics["run_name"].nunique()),
        "num_validation_examples": int(summary["example_id"].nunique()),
        "non_unanimous_rate": float((summary["disagreement_count"] > 0).mean()),
        "accuracy_mean": float(metrics["eval_accuracy"].mean()),
        "accuracy_min": float(metrics["eval_accuracy"].min()),
        "accuracy_max": float(metrics["eval_accuracy"].max()),
        "accuracy_std": float(metrics["eval_accuracy"].std(ddof=0)),
        "mean_pairwise_agreement": float(np.nanmean(upper_triangle.to_numpy())),
        "best_model_by_agreement": {
            "run_name": best_by_agreement["run_name"],
            "agreement_score": float(best_by_agreement["agreement_score"]),
            "eval_accuracy": float(best_by_agreement["eval_accuracy"]),
            "is_top_3_by_accuracy": bool(best_by_agreement["run_name"] in top_3_by_accuracy),
        },
        "agreement_accuracy_correlation": {
            "pearson_r": float(pearson_r),
            "p_value": float(pearson_p),
        },
        "group_comparison": group_comparison.to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_validation_predictions(args.artifacts_root)
    metrics = load_run_metrics(args.artifacts_root)
    sentence_summary = build_sentence_summary(predictions)
    pairwise_agreement = build_pairwise_agreement(predictions)
    model_scores = build_model_scores(predictions, metrics)
    stable, unstable = summarize_groups(sentence_summary, args.top_disagreement_quantile)
    group_comparison = compare_groups(stable, unstable)

    sentence_summary.to_csv(args.output_dir / "sentence_summary.csv", index=False)
    pairwise_agreement.to_csv(args.output_dir / "pairwise_agreement.csv")
    model_scores.to_csv(args.output_dir / "model_scores.csv", index=False)
    group_comparison.to_csv(args.output_dir / "stable_vs_unstable_comparison.csv", index=False)

    top_unstable = unstable.sort_values(["disagreement_count", "confidence_std"], ascending=[False, False]).head(50)
    top_unstable.to_csv(args.output_dir / "top_unstable_sentences.csv", index=False)

    create_plots(sentence_summary, pairwise_agreement, model_scores, args.output_dir)

    report = build_summary_report(sentence_summary, pairwise_agreement, metrics, model_scores, group_comparison)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
