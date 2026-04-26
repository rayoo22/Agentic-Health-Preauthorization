import argparse
import json
from pathlib import Path

import pandas as pd


STAGE_COLUMNS = [
    "match_extraction_success",
    "match_member_id",
    "match_diagnosis",
    "match_requested_service",
    "match_claim_amount",
    "match_member_exists",
    "match_balance_sufficient",
    "match_final_decision",
]


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_accuracy(series):
    normalized = series.map(normalize_text).str.upper()
    valid = normalized[normalized.isin(["TRUE", "FALSE"])]
    if len(valid) == 0:
        return None
    return float((valid == "TRUE").mean())


def direct_accuracy(expected, actual):
    expected = expected.map(normalize_text)
    actual = actual.map(normalize_text)
    valid = (expected != "") & (actual != "")
    if not valid.any():
        return None
    return float((expected[valid] == actual[valid]).mean())


def confusion_matrix_from_series(expected, actual):
    expected = expected.map(normalize_text)
    actual = actual.map(normalize_text)
    labels = sorted({label for label in set(expected) | set(actual) if label})
    matrix = {label: {other: 0 for other in labels} for label in labels}

    for exp, act in zip(expected, actual):
        if not exp or not act:
            continue
        matrix[exp][act] = matrix[exp].get(act, 0) + 1

    return labels, matrix


def binary_metrics(expected, actual, positive_label):
    expected = expected.map(normalize_text)
    actual = actual.map(normalize_text)

    tp = int(((expected == positive_label) & (actual == positive_label)).sum())
    fp = int(((expected != positive_label) & (actual == positive_label)).sum())
    fn = int(((expected == positive_label) & (actual != positive_label)).sum())
    tn = int(((expected != positive_label) & (actual != positive_label)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(expected) if len(expected) else 0.0

    return {
        "label": positive_label,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_per_class_accuracy(df):
    rows = []
    for category, group in df.groupby("category"):
        rows.append(
            {
                "category": category,
                "final_decision_accuracy": direct_accuracy(
                    group["expected_final_decision"],
                    group["actual_final_decision"],
                ),
                "end_to_end_accuracy": float(
                    group[STAGE_COLUMNS]
                    .apply(lambda row: all(value == "TRUE" or value == "" for value in row), axis=1)
                    .mean()
                ),
                "cases": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("category")


def compute_stage_success_rates(df):
    rows = []
    for column in STAGE_COLUMNS:
        rows.append(
            {
                "stage": column.replace("match_", ""),
                "success_rate": safe_accuracy(df[column]),
            }
        )
    return pd.DataFrame(rows)


def save_confusion_matrix_plot(confusion_df, output_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(confusion_df.values, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(confusion_df.columns)))
    ax.set_yticks(range(len(confusion_df.index)))
    ax.set_xticklabels(confusion_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(confusion_df.index)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Expected label")
    ax.set_title("Confusion Matrix")

    for row_index in range(len(confusion_df.index)):
        for col_index in range(len(confusion_df.columns)):
            ax.text(
                col_index,
                row_index,
                str(int(confusion_df.iat[row_index, col_index])),
                ha="center",
                va="center",
                color="black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Compute quantitative metrics from offline workflow results.")
    parser.add_argument("--input", required=True, help="Path to the results CSV file.")
    parser.add_argument("--output-dir", default="test_data/metrics", help="Directory to save metric outputs.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    expected_decision = df["expected_final_decision"].fillna("UNKNOWN")
    actual_decision = df["actual_final_decision"].fillna("UNKNOWN")
    overall_accuracy = direct_accuracy(expected_decision, actual_decision)
    stage_success = compute_stage_success_rates(df)
    per_class_accuracy = compute_per_class_accuracy(df)

    approved_metrics = binary_metrics(expected_decision, actual_decision, "APPROVED")
    denied_metrics = binary_metrics(expected_decision, actual_decision, "DENIED")
    labels, confusion = confusion_matrix_from_series(expected_decision, actual_decision)
    confusion_df = pd.DataFrame.from_dict(confusion, orient="index")
    confusion_df.index.name = "expected_label"
    confusion_df = confusion_df.reindex(index=labels, columns=labels, fill_value=0)

    summary = {
        "overall_final_decision_accuracy": overall_accuracy,
        "approved_metrics": approved_metrics,
        "denied_metrics": denied_metrics,
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion,
    }

    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    stage_success.to_csv(output_dir / "stage_success_rates.csv", index=False)
    per_class_accuracy.to_csv(output_dir / "per_class_accuracy.csv", index=False)
    confusion_df.to_csv(output_dir / "confusion_matrix.csv")
    plot_saved = save_confusion_matrix_plot(confusion_df, output_dir / "confusion_matrix.png")

    print("Overall final decision accuracy:", overall_accuracy)
    print("\nApproved metrics:")
    print(json.dumps(approved_metrics, indent=2))
    print("\nDenied metrics:")
    print(json.dumps(denied_metrics, indent=2))
    print("\nConfusion matrix:")
    print(confusion_df.to_string())
    if plot_saved:
        print(f"\nSaved confusion matrix plot to {output_dir / 'confusion_matrix.png'}")
    else:
        print("\nConfusion matrix plot was not generated because matplotlib is not installed.")
    print(f"\nSaved metric outputs to {output_dir}")


if __name__ == "__main__":
    main()
