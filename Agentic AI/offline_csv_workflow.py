import argparse
import csv
import random
import os
from pathlib import Path

from dotenv import load_dotenv

from members_db import MembersDB
from openai_agent import OpenAIEmailAgent
from RAG.query_policy_rag import query_policy_rag, format_policy_context


REQUIRED_EXTRACTION_FIELDS = ("member_id", "diagnosis", "requested_service", "claim_amount")


def normalize_amount(value):
    # Convert amount-like values into floats so comparisons work consistently.
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def values_match(expected, actual):
    # Compare expected vs actual values and return TRUE/FALSE for evaluation columns.
    if expected in (None, "", "UNKNOWN"):
        return ""

    expected_text = str(expected).strip()
    actual_text = "" if actual is None else str(actual).strip()

    if expected_text.upper() in {"TRUE", "FALSE"}:
        return str(expected_text.upper() == actual_text.upper()).upper()

    expected_amount = normalize_amount(expected_text)
    actual_amount = normalize_amount(actual_text)
    if expected_amount is not None and actual_amount is not None:
        return str(expected_amount == actual_amount).upper()

    return str(expected_text == actual_text).upper()


def run_case(row, agent, members_db):
    # This is the core offline pipeline for one CSV row.
    # It mirrors the main workflow, but uses subject/body from CSV instead of Gmail.
    subject = row.get("subject", "")
    body = row.get("body", "")

    # Step 1: run LLM-based extraction on the synthetic email text.
    claim_data = agent.extract_claim_data(subject, body)

    result = {
        "actual_extraction_success": "TRUE" if claim_data else "FALSE",
        "actual_member_id": "",
        "actual_diagnosis": "",
        "actual_requested_service": "",
        "actual_claim_amount": "",
        "actual_member_exists": "",
        "actual_balance_sufficient": "",
        "actual_final_decision": "",
        "actual_reason": "",
    }

    if not claim_data:
        result["actual_final_decision"] = "EXTRACTION_FAILED"
        result["actual_reason"] = "Could not extract claim data"
        return result

    missing_fields = claim_data.get("missing_fields") or []
    ambiguity_flags = claim_data.get("ambiguity_flags") or []
    blocking_reasons = []

    for field in REQUIRED_EXTRACTION_FIELDS:
        if claim_data.get(field) in (None, "") and field not in missing_fields:
            missing_fields.append(field)

    for field in missing_fields:
        blocking_reasons.append(f"{field.replace('_', ' ').capitalize()} missing")

    for field in ambiguity_flags:
        if field in {"diagnosis", "requested_service"}:
            blocking_reasons.append(f"{field.replace('_', ' ').capitalize()} ambiguous")

    # Step 2: save extracted fields into the output row for later evaluation.
    result["actual_member_id"] = claim_data.get("member_id", "")
    result["actual_diagnosis"] = claim_data.get("diagnosis", "")
    result["actual_requested_service"] = claim_data.get("requested_service", "")
    result["actual_claim_amount"] = claim_data.get("claim_amount", "")

    if blocking_reasons:
        result["actual_final_decision"] = "EXTRACTION_FAILED"
        result["actual_reason"] = "; ".join(dict.fromkeys(blocking_reasons))
        return result

    # Step 3: perform rule-based member validation using the database.
    member_id = claim_data.get("member_id")
    member = members_db.get_member(member_id) if member_id else None
    result["actual_member_exists"] = "TRUE" if member else "FALSE"

    if not member:
        result["actual_final_decision"] = "DENIED"
        result["actual_reason"] = "Member not found in system"
        return result

    claim_amount = normalize_amount(claim_data.get("claim_amount"))
    if claim_amount is None:
        result["actual_final_decision"] = "EXTRACTION_FAILED"
        result["actual_reason"] = "Claim amount missing or invalid"
        return result

    # Step 4: apply the deterministic balance eligibility rule.
    balance_sufficient = member["policy_balance"] > claim_amount
    result["actual_balance_sufficient"] = "TRUE" if balance_sufficient else "FALSE"

    if not balance_sufficient:
        result["actual_final_decision"] = "DENIED"
        result["actual_reason"] = (
            f'Insufficient policy balance: Ksh. {member["policy_balance"]} available, '
            f"Ksh. {claim_amount} required"
        )
        return result

    # Step 5: retrieve policy context through the RAG layer.
    relevant_chunks = query_policy_rag(
        claim_data.get("diagnosis", ""),
        claim_data.get("requested_service", ""),
    )
    policy_context = format_policy_context(relevant_chunks)

    # Step 6: run clinical adjudication using the retrieved policy context.
    adjudication = agent.clinical_adjudication(
        claim_data.get("diagnosis", ""),
        claim_data.get("requested_service", ""),
        policy_context,
    )

    result["actual_final_decision"] = adjudication.get("decision", "PENDING")
    result["actual_reason"] = adjudication.get("reasoning", "")
    return result


def build_output_row(row, result):
    # Merge the original labeled row with actual outputs and match indicators.
    output = dict(row)
    output.update(result)

    output["match_member_id"] = values_match(row.get("expected_member_id"), result["actual_member_id"])
    output["match_diagnosis"] = values_match(row.get("expected_diagnosis"), result["actual_diagnosis"])
    output["match_requested_service"] = values_match(
        row.get("expected_requested_service"), result["actual_requested_service"]
    )
    output["match_claim_amount"] = values_match(
        row.get("expected_claim_amount"), result["actual_claim_amount"]
    )
    output["match_extraction_success"] = values_match(
        row.get("expected_extraction_success"), result["actual_extraction_success"]
    )
    output["match_member_exists"] = values_match(
        row.get("expected_member_exists"), result["actual_member_exists"]
    )
    output["match_balance_sufficient"] = values_match(
        row.get("expected_balance_sufficient"), result["actual_balance_sufficient"]
    )
    output["match_final_decision"] = values_match(
        row.get("expected_final_decision"), result["actual_final_decision"]
    )

    return output


def process_csv(input_csv: Path, output_csv: Path, shuffle: bool = False, seed: int | None = None):
    load_dotenv()

    # Load the benchmark dataset from CSV.
    with input_csv.open("r", encoding="utf-8-sig", newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        rows = list(reader)

    if not rows:
        raise ValueError("Input CSV contains no data rows.")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    # Create the components reused across all cases.
    agent = OpenAIEmailAgent()
    members_db = MembersDB()

    try:
        output_rows = []
        for row in rows:
            # Process one case at a time, then append its evaluated output.
            result = run_case(row, agent, members_db)
            output_rows.append(build_output_row(row, result))
    finally:
        # Always close the DB connection, even if one case fails.
        members_db.close()

    # Write the full evaluated dataset back to disk.
    fieldnames = list(output_rows[0].keys())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Processed {len(output_rows)} cases into {output_csv}")


def parse_args():
    # Read file locations and optional shuffle settings from CLI or .env defaults.
    parser = argparse.ArgumentParser(description="Offline CSV ingestion workflow for synthetic claim emails.")
    parser.add_argument(
        "--input",
        default=os.getenv("OFFLINE_CSV_INPUT", "test_data/hundred_mixed_emails.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("OFFLINE_CSV_OUTPUT", "test_data/hundred_mixed_results.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=os.getenv("OFFLINE_CSV_SHUFFLE", "false").lower() == "true",
        help="Shuffle dataset rows before processing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("OFFLINE_CSV_SEED")) if os.getenv("OFFLINE_CSV_SEED") else None,
        help="Optional random seed for reproducible shuffling.",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    # Entry point: resolve config and run the offline benchmark.
    process_csv(
        Path(args.input),
        Path(args.output),
        shuffle=args.shuffle,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
