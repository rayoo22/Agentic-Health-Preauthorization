import argparse
import base64
import csv
import mimetypes
import os
import pickle
import random
import time
from email import policy
from email.parser import BytesParser
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def authenticate():
    # Reuse an existing Gmail OAuth session when possible.
    creds = None

    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token_file:
            creds = pickle.load(token_file)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                if os.path.exists("token.pickle"):
                    os.remove("token.pickle")
                creds = None

        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file("agenticai_credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.pickle", "wb") as token_file:
            pickle.dump(creds, token_file)

    return build("gmail", "v1", credentials=creds)


def parse_eml(eml_path: Path):
    # Parse a saved .eml file into a standard email object.
    with eml_path.open("rb") as handle:
        return BytesParser(policy=policy.default).parse(handle)


def extract_text_body(message):
    # Prefer the plain-text body so the sent message matches the synthetic input cleanly.
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain" and not part.get_filename():
                return part.get_content()
        return ""
    return message.get_content()


def build_outgoing_message(source_message, recipient, preserve_subject_prefix=False):
    # Build a new outbound Gmail message from the stored .eml content.
    msg = EmailMessage()
    subject = source_message.get("Subject", "Synthetic Claim Email")
    if preserve_subject_prefix and not subject.lower().startswith("fwd:"):
        subject = f"Fwd: {subject}"

    msg["To"] = recipient
    msg["Subject"] = subject

    original_from = source_message.get("From")
    original_message_id = source_message.get("Message-ID")
    if original_from:
        msg["Reply-To"] = original_from
    if original_message_id:
        msg["X-Original-Message-ID"] = original_message_id

    body = extract_text_body(source_message)
    msg.set_content(body)

    for part in source_message.iter_attachments():
        payload = part.get_payload(decode=True)
        if payload is None:
            continue

        filename = part.get_filename() or "attachment"
        content_type = part.get_content_type()
        maintype, subtype = content_type.split("/", 1) if "/" in content_type else ("application", "octet-stream")

        if content_type == "application/octet-stream":
            guessed_type, _ = mimetypes.guess_type(filename)
            if guessed_type and "/" in guessed_type:
                maintype, subtype = guessed_type.split("/", 1)

        msg.add_attachment(payload, maintype=maintype, subtype=subtype, filename=filename)

    return msg


def send_message(service, message):
    # Gmail API expects the MIME message to be base64-url encoded.
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    payload = {"raw": raw}
    return service.users().messages().send(userId="me", body=payload).execute()


def iter_eml_files(input_path: Path):
    # Accept either a single .eml file or a whole folder of .eml files.
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.eml"))


def load_categories(metadata_csv: Path):
    # Load email_id -> category so we can mix sends across classes.
    mapping = {}
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            email_id = (row.get("email_id") or "").strip()
            category = (row.get("category") or "").strip()
            if email_id:
                mapping[email_id] = category
    return mapping


def mix_files_by_category(eml_files, metadata_csv: Path, seed=None):
    # Interleave categories so consecutive sends are not all from the same class.
    categories = load_categories(metadata_csv)
    grouped = {}

    for eml_file in eml_files:
        category = categories.get(eml_file.stem, "uncategorized")
        grouped.setdefault(category, []).append(eml_file)

    rng = random.Random(seed)
    category_names = sorted(grouped.keys())
    for category in category_names:
        rng.shuffle(grouped[category])

    mixed = []
    while any(grouped[category] for category in category_names):
        for category in category_names:
            if grouped[category]:
                mixed.append(grouped[category].pop(0))

    return mixed


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Send .eml files to a Gmail inbox.")
    parser.add_argument(
        "--input",
        default=os.getenv("SEND_EML_INPUT"),
        help="Path to a .eml file or a folder of .eml files.",
    )
    parser.add_argument(
        "--to",
        default=os.getenv("SEND_EML_TO"),
        help="Destination Gmail address.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=float(os.getenv("SEND_EML_DELAY_SECONDS", "0")),
        help="Optional delay between sends.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle emails before sending.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("SEND_EML_SEED")) if os.getenv("SEND_EML_SEED") else None,
        help="Optional random seed for reproducible mixing.",
    )
    parser.add_argument(
        "--mix-by-category",
        action="store_true",
        help="Interleave emails by category using metadata CSV.",
    )
    parser.add_argument(
        "--metadata-csv",
        default=os.getenv("SEND_EML_METADATA_CSV", "test_data/hundred_mixed_emails.csv"),
        help="CSV file containing email_id and category columns.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("SEND_EML_LIMIT")) if os.getenv("SEND_EML_LIMIT") else None,
        help="Optional max number of emails to send.",
    )
    parser.add_argument(
        "--preserve-subject-prefix",
        action="store_true",
        help="Prefix the sent subject with 'Fwd:' to make test emails easier to spot.",
    )
    args = parser.parse_args()

    if not args.to:
        raise ValueError("No destination Gmail address provided. Set SEND_EML_TO in .env or pass --to.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Load available .eml files first, then optionally randomize the send order.
    eml_files = iter_eml_files(input_path)
    if not eml_files:
        raise ValueError("No .eml files found to send.")

    if args.mix_by_category:
        metadata_csv = Path(args.metadata_csv)
        if not metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
        eml_files = mix_files_by_category(eml_files, metadata_csv, seed=args.seed)
    elif args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(eml_files)

    if args.limit is not None:
        eml_files = eml_files[: args.limit]

    service = authenticate()

    sent_count = 0
    for eml_file in eml_files:
        # Parse each synthetic email and resend it through the authenticated Gmail account.
        source_message = parse_eml(eml_file)
        outgoing = build_outgoing_message(
            source_message,
            args.to,
            preserve_subject_prefix=args.preserve_subject_prefix,
        )
        result = send_message(service, outgoing)
        sent_count += 1
        print(f"Sent {eml_file.name} -> {args.to} (Gmail ID: {result['id']})")

        if args.delay_seconds > 0 and sent_count < len(eml_files):
            time.sleep(args.delay_seconds)

    print(f"Finished sending {sent_count} email(s).")


if __name__ == "__main__":
    main()
