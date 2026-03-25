import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


class VertexClaimJudge:
    def __init__(self, model=None, project=None, location=None):
        load_dotenv()

        self.model = model or os.getenv("VERTEX_JUDGE_MODEL")
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION")

        if not self.project:
            raise ValueError("GOOGLE_CLOUD_PROJECT is not set.")

        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
        os.environ["GOOGLE_CLOUD_PROJECT"] = self.project
        os.environ["GOOGLE_CLOUD_LOCATION"] = self.location

        self.client = genai.Client()

    def judge_claim_decision(
        self,
        email_subject,
        email_body,
        claim_data,
        member_info,
        policy_context,
        proposed_decision,
        proposed_reasoning,
    ):
        prompt_payload = make_json_safe({
            "email_subject": email_subject,
            "email_body": email_body,
            "claim_data": claim_data,
            "member_info": member_info,
            "policy_context": policy_context,
            "proposed_decision": proposed_decision,
            "proposed_reasoning": proposed_reasoning,
        })

        system_instruction = (
            "You are an independent healthcare claims quality judge. "
            "Review the proposed claim decision using the email, extracted data, "
            "member details, and policy context. "
            "Check extraction consistency, business-rule correctness, and policy alignment. "
            "Return only valid JSON."
        )

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "judge_agrees": {"type": "BOOLEAN"},
                "judge_verdict": {"type": "STRING"},
                "judge_score": {"type": "NUMBER"},
                "judge_reasoning": {"type": "STRING"},
                "stage_evaluation": {
                    "type": "OBJECT",
                    "properties": {
                        "extraction": {
                            "type": "OBJECT",
                            "properties": {
                                "score": {"type": "NUMBER"},
                                "assessment": {"type": "STRING"},
                            },
                            "required": ["score", "assessment"],
                        },
                        "member_validation": {
                            "type": "OBJECT",
                            "properties": {
                                "score": {"type": "NUMBER"},
                                "assessment": {"type": "STRING"},
                            },
                            "required": ["score", "assessment"],
                        },
                        "balance_check": {
                            "type": "OBJECT",
                            "properties": {
                                "score": {"type": "NUMBER"},
                                "assessment": {"type": "STRING"},
                            },
                            "required": ["score", "assessment"],
                        },
                        "policy_alignment": {
                            "type": "OBJECT",
                            "properties": {
                                "score": {"type": "NUMBER"},
                                "assessment": {"type": "STRING"},
                            },
                            "required": ["score", "assessment"],
                        },
                        "final_decision": {
                            "type": "OBJECT",
                            "properties": {
                                "score": {"type": "NUMBER"},
                                "assessment": {"type": "STRING"},
                            },
                            "required": ["score", "assessment"],
                        },
                    },
                    "required": [
                        "extraction",
                        "member_validation",
                        "balance_check",
                        "policy_alignment",
                        "final_decision",
                    ],
                },
                "judge_flags": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
            },
            "required": [
                "judge_agrees",
                "judge_verdict",
                "judge_score",
                "judge_reasoning",
                "stage_evaluation",
                "judge_flags",
            ],
        }

        response = self.client.models.generate_content(
            model=self.model,
            contents=json.dumps(prompt_payload, ensure_ascii=True, indent=2),
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0,
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )

        text = (response.text or "").strip()
        return json.loads(text)
