# Agentic AI Healthcare Claims Processing System

Project Repository: https://github.com/rayoo22/Agentic-Health-Preauthorization

This project automates healthcare claim intake and adjudication from email. It reads claim emails, extracts structured claim data with an LLM, validates member and balance information from PostgreSQL, retrieves policy context with RAG, makes a claim decision, sends a response email, and can optionally send a second judge report using Vertex AI.

The current workflow is intentionally conservative. If critical claim fields are missing, vague, or ambiguous, the system flags the issue and can stop the claim before downstream adjudication rather than forcing a weakly supported decision.

## What The System Does

The main workflow is built around Gmail:

1. Fetch unread emails from a Gmail label.
2. Extract:
   - `member_id`
   - `diagnosis`
   - `requested_service`
   - `claim_amount`
3. Validate extracted output and flag:
   - missing fields
   - ambiguous fields
   - invalid claim amounts
4. Check whether the member exists.
5. Check whether the member has enough policy balance.
6. Retrieve relevant policy sections from the local RAG index.
7. Run clinical adjudication with policy context.
8. Store and update claim records in PostgreSQL.
9. Send a claim decision email.
10. Optionally send a detailed Vertex AI judge evaluation in the same Gmail thread.

The repo also includes an offline CSV workflow for evaluation, plus tooling for generating and sending synthetic `.eml` claim emails.

## Main Workflows

### 1. Gmail Workflow

Entry point: `Agentic AI/main.py`

This is the live end-to-end workflow. It:

- reads unread claim emails from Gmail
- processes each claim through extraction, validation, RAG, and adjudication
- stops incomplete or ambiguous claims before adjudication when critical information is missing
- sends a decision response
- marks the original email as read
- optionally sends a Vertex AI judge report in the same thread

### 2. Offline CSV Workflow

Entry point: `Agentic AI/offline_csv_workflow.py`

This workflow is for evaluation and experimentation. It:

- reads claim inputs from a CSV instead of Gmail
- runs extraction, output validation, member lookup, balance check, RAG, and adjudication
- writes results to a CSV
- compares outputs with expected labels if those columns are present
- supports per-class, stage-wise, and final-decision evaluation

It does not:

- fetch from Gmail
- send emails
- insert claims into the claims table
- deduct balances

## Project Structure

### Core Application

- `Agentic AI/main.py`
  Main Gmail-based claims workflow.

- `Agentic AI/openai_agent.py`
  OpenAI-powered extraction, adjudication, and response generation.

- `Agentic AI/gmail_reader.py`
  Gmail API authentication, inbox reads, replies, and sending standalone emails.

### Database Access

- `Agentic AI/db_manager.py`
  Email metadata storage and legacy claim-related DB helpers.

- `Agentic AI/claims_db.py`
  Claim insert/update operations.

- `Agentic AI/members_db.py`
  Member lookup and balance deduction.

- `Agentic AI/policies_db.py`
  Policy lookup from PostgreSQL.

### RAG Pipeline

- `Agentic AI/setup_rag.py`
  One-time RAG index build script.

- `Agentic AI/RAG/read_policy_documents.py`
  Reads policy markdown documents.

- `Agentic AI/RAG/chunking_documents.py`
  Chunks policy text.

- `Agentic AI/RAG/generate_embeddings.py`
  Generates embeddings and builds FAISS index.

- `Agentic AI/RAG/query_policy_rag.py`
  Retrieves the most relevant policy chunks for a claim.

### Prompt Files

- `Agentic AI/prompts/extract_claim_data_prompt.txt`
- `Agentic AI/prompts/clinical_adjudication_prompt.txt`
- `Agentic AI/prompts/generate_claim_response_email.txt`

The extraction and adjudication prompts use conservative instructions and few-shot examples so the models:

- return structured JSON only
- avoid guessing unsupported claim details
- preserve missingness and ambiguity explicitly
- deny or stop claims when critical information is too weak for safe adjudication

### Vertex AI Judge

- `Agentic AI/vertex_judge.py`
  A second-pass LLM judge using Vertex AI. It reviews the original email and the workflow output and returns:
  - overall agreement
  - overall verdict
  - overall score
  - reasoning
  - per-stage evaluation for:
    - extraction
    - member validation
    - balance check
    - policy alignment
    - final decision

### Evaluation And Synthetic Test Data

- `Agentic AI/offline_csv_workflow.py`
  Offline evaluation workflow.

- `Agentic AI/evaluate_results.py`
  Summarizes results per class and per metric.

- `Agentic AI/compute_metrics.py`
  Computes overall decision metrics, per-class accuracy, stage success rates, and a confusion matrix.

- `Agentic AI/evaluation_notebook.ipynb`
  Notebook for analyzing saved results only. It does not rerun the LLM.

- `Agentic AI/send_eml_to_gmail.py`
  Sends generated `.eml` files into a Gmail inbox for end-to-end testing.

- `Agentic AI/test_data/`
  Contains:
  - CSV datasets
  - generated `.eml` files
  - evaluation outputs
  - templates and helpers

## Data Flow

```text
Gmail Inbox
  -> Email Body Extraction
  -> OpenAI Claim Extraction
  -> Extraction Validation / Gating
  -> Member Lookup / Balance Check
  -> RAG Policy Retrieval
  -> OpenAI Clinical Adjudication
  -> Claim Decision Email
  -> Optional Vertex AI Judge Email
```

Offline evaluation uses the same middle stages, but replaces Gmail input with CSV rows and writes comparable benchmark outputs to disk for later analysis.

## Requirements

- Python 3.10+
- PostgreSQL
- Gmail API credentials
- OpenAI API access
- FAISS
- Optional: Vertex AI access for judge reports

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Git Bash:

```bash
source .venv/Scripts/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install faiss-cpu google-genai
```

### 3. Configure environment variables

Create `Agentic AI/.env` with values like:

```env
DB_HOST=localhost
DB_PORT=5462
DB_NAME=email_db
DB_USER=postgres
DB_PASSWORD=your_password

MAX_EMAILS=50
TOP_K=5

OPENAI_API_KEY=your_openai_key
OPEN_AI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-ada-002

SEND_EML_INPUT=test_data/emails
SEND_EML_TO=your_test_gmail@gmail.com
SEND_EML_DELAY_SECONDS=0
SEND_EML_METADATA_CSV=test_data/hundred_mixed_emails.csv
SEND_EML_SEED=42

OFFLINE_CSV_INPUT=test_data/hundred_mixed_emails.csv
OFFLINE_CSV_OUTPUT=test_data/hundred_mixed_results.csv

ENABLE_VERTEX_JUDGE=false
JUDGE_REPORT_EMAIL=your_email@gmail.com
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
VERTEX_JUDGE_MODEL=gemini-2.5-flash
```

### 4. Configure Gmail API

Place your Gmail OAuth desktop credential file in `Agentic AI/`.

Current code expects:

- `credentials.json` for the main Gmail workflow
- `agenticai_credentials.json` for the synthetic email sender

The first run will create a `token.pickle` after OAuth login.

### 5. Build the RAG index

From `Agentic AI/`:

```bash
python setup_rag.py
```

This generates:

- `policy_index.faiss`
- `chunks_metadata.pkl`

## Running The System

### Gmail Workflow

From `Agentic AI/`:

```bash
python main.py
```

What it does:

- fetches unread emails from the `Agentic_AI` label
- processes each claim
- sends a decision reply
- optionally sends a Vertex judge report

### Offline Evaluation Workflow

If `.env` contains `OFFLINE_CSV_INPUT` and `OFFLINE_CSV_OUTPUT`, run:

```bash
python offline_csv_workflow.py
```

Optional:

```bash
python offline_csv_workflow.py --shuffle --seed 42
```

What it does:

- reads subject/body pairs from the CSV benchmark
- runs the same extraction, validation, retrieval, and adjudication logic used in the workflow core
- writes `actual_*` outputs and `match_*` evaluation columns to the output CSV
- can return `EXTRACTION_FAILED` when critical information is missing or ambiguous

### Evaluate Saved Offline Results

```bash
python evaluate_results.py --input test_data/hundred_mixed_results.csv --output test_data/hundred_mixed_summary.csv
```

### Compute Overall Metrics

```bash
python compute_metrics.py --input test_data/hundred_mixed_results.csv
```

This writes metric outputs to `test_data/metrics/`, including:

- `summary_metrics.json`
- `stage_success_rates.csv`
- `per_class_accuracy.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png` if `matplotlib` is installed

### Analyze Results In Notebook

Open:

- `Agentic AI/evaluation_notebook.ipynb`

The notebook reads saved CSV outputs and produces tables and charts. It does not call the models again.

### Send Synthetic Emails To Gmail

If `.env` contains `SEND_EML_INPUT` and `SEND_EML_TO`, run:

```bash
python send_eml_to_gmail.py
```

Optional sending behavior:

```bash
python send_eml_to_gmail.py --shuffle --seed 42
python send_eml_to_gmail.py --mix-by-category --seed 42
```

## Evaluation Approach

The best evaluation strategy in this repo is:

1. Use the offline CSV workflow as the main benchmark.
2. Use labeled datasets with expected outputs.
3. Evaluate stage-by-stage:
   - extraction
   - extraction validation / gating
   - member existence
   - balance sufficiency
   - final decision
4. Evaluate per class:
   - `clean_valid`
   - `missing_information`
   - `ambiguous_wording`
   - `member_not_found`
   - `insufficient_balance`
   - `policy_exclusion_or_weak_medical_necessity`
5. Use the Gmail workflow for smaller end-to-end validation.

## Vertex AI Judge

If enabled, the judge reviews the original incoming email plus the workflow output and sends a detailed evaluation report. It is currently used as a reviewer, not as the final decision-maker.

To enable it:

```env
ENABLE_VERTEX_JUDGE=true
JUDGE_REPORT_EMAIL=your_email@gmail.com
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
VERTEX_JUDGE_MODEL=gemini-2.5-flash
```

You also need Google Application Default Credentials. For local development:

```bash
gcloud auth application-default login
```

## Notes

- The Gmail workflow is the most complete production-style path.
- The offline workflow is the best place to benchmark extraction and reasoning.
- The offline workflow now reflects the conservative system behavior used in the dissertation write-up.
- The synthetic email sender is useful for realistic Gmail inbox testing.
- Vertex judge failures do not stop the main claim-processing workflow.

## Security

- Do not commit real API keys, OAuth tokens, or database passwords.
- Keep `.env`, credential JSON files, and `token.pickle` secure.
- Use test inboxes and test databases during evaluation.
