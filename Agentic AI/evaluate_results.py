import argparse
import csv
from pathlib import Path


MATCH_COLUMNS = [
    "match_extraction_success",
    "match_member_id",
    "match_diagnosis",
    "match_requested_service",
    "match_claim_amount",
    "match_member_exists",
    "match_balance_sufficient",
    "match_final_decision",
]


def accuracy(rows, column):
    relevant = [row for row in rows if row.get(column, "") in {"TRUE", "FALSE"}]
    if not relevant:
        return ""
    correct = sum(1 for row in relevant if row[column] == "TRUE")
    return f"{correct}/{len(relevant)} ({(correct / len(relevant)) * 100:.1f}%)"


def summarize(rows):
    groups = {}
    for row in rows:
        category = row.get("category", "uncategorized")
        groups.setdefault(category, []).append(row)

    summary_rows = []
    for category, category_rows in sorted(groups.items()):
        summary = {
            "category": category,
            "cases": str(len(category_rows)),
        }
        for column in MATCH_COLUMNS:
            summary[column] = accuracy(category_rows, column)
        summary_rows.append(summary)

    overall = {"category": "OVERALL", "cases": str(len(rows))}
    for column in MATCH_COLUMNS:
        overall[column] = accuracy(rows, column)
    summary_rows.append(overall)

    return summary_rows


def print_summary(summary_rows):
    for row in summary_rows:
        print(f"\n[{row['category']}] cases={row['cases']}")
        for column in MATCH_COLUMNS:
            value = row.get(column, "")
            if value:
                print(f"  {column}: {value}")


def write_summary(summary_rows, output_csv: Path):
    fieldnames = ["category", "cases", *MATCH_COLUMNS]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize offline workflow results by class.")
    parser.add_argument("--input", required=True, help="Path to results CSV from offline workflow.")
    parser.add_argument("--output", help="Optional path to write summary CSV.")
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("Results CSV contains no data rows.")

    summary_rows = summarize(rows)
    print_summary(summary_rows)

    if args.output:
        write_summary(summary_rows, Path(args.output))


if __name__ == "__main__":
    main()
