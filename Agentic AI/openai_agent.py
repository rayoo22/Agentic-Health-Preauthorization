import openai
import os
import json 
from pathlib import Path


REQUIRED_CLAIM_FIELDS = ("member_id", "diagnosis", "requested_service", "claim_amount")
VAGUE_TEXT_VALUES = {
    "",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "not provided",
    "not specified",
    "unspecified",
    "medical treatment",
    "treatment",
    "service",
}


class OpenAIEmailAgent:
    """AI agent for processing healthcare claims using OpenAI GPT models
    
    This class handles:
    - Extracting structured claim data from unstructured email content
    - Performing clinical adjudication based on medical necessity
    - Generating professional response emails for claim decisions
    """
    
    def __init__(self, model='gpt-4o'):
        """Initialize OpenAI agent with API key and model configuration
        
        Args:
            model (str): OpenAI model to use (default: gpt-4o)
        """
        # Set OpenAI API key from environment variable
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.model = model
    
    def generate_email_response(self, subject, body, sender, attachments=None):
        """Generate a professional email response (legacy method - kept for compatibility)
        
        Args:
            subject (str): Original email subject
            body (str): Original email body content
            sender (str): Email sender address
            attachments (list): List of email attachments (optional)
            
        Returns:
            str: Generated professional email response
        """
        # Construct prompt for general email response generation
        prompt = f"""You are an AI email assistant. Analyze the following email and generate a professional response.

Email Details:
From: {sender}
Subject: {subject}

Body:
{body}

{f"Attachments: {len(attachments)} file(s)" if attachments else "No attachments"}

Generate a professional, concise email response addressing the key points."""

        # Call OpenAI API to generate response
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional email assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def extract_claim_data(self, subject, body):
        """Extract structured claim information from unstructured email content
        
        This method uses GPT to parse email content and extract:
        - Member ID
        - Medical diagnosis
        - Requested healthcare service
        - Claim amount
        
        Args:
            subject (str): Email subject line
            body (str): Email body content
            
        Returns:
            dict: Structured claim data or None if extraction fails
        """
        # Create specific prompt for claim data extraction
        prompt = Path("prompts/extract_claim_data_prompt.txt")
        extract_claim_data_prompt = prompt.read_text()\
                                    .replace("{subject}", subject)\
                                    .replace("{body}", body)
        """Extract claim information from this email. Return ONLY a JSON object with these fields:
- member_id
- diagnosis
- requested_service
- claim_amount (numeric value only)

Email Subject: {subject}
Email Body: {body}

Return JSON only, no explanation."""

        # Call OpenAI API with structured data extraction prompt
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You extract structured data from emails. Return only valid JSON."},
                {"role": "user", "content": extract_claim_data_prompt}
            ],
            max_tokens=500
        )
        
        try:
            # Clean and parse JSON response from GPT
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code block formatting if present
            if content.startswith('```json'):
                content = content[7:-3].strip()
            elif content.startswith('```'):
                content = content[3:-3].strip()
                
            # Parse JSON and return structured data
            claim_data = json.loads(content)
            return self._validate_claim_data(claim_data)
        except:
            # Return None if JSON parsing fails
            return None

    def _validate_claim_data(self, claim_data):
        """Validate extracted claim data and mark incomplete or vague fields."""
        if not isinstance(claim_data, dict):
            return None

        normalized = {
            "member_id": claim_data.get("member_id"),
            "diagnosis": claim_data.get("diagnosis"),
            "requested_service": claim_data.get("requested_service"),
            "claim_amount": claim_data.get("claim_amount"),
            "missing_fields": claim_data.get("missing_fields") or [],
            "ambiguity_flags": claim_data.get("ambiguity_flags") or [],
        }

        if not isinstance(normalized["missing_fields"], list):
            normalized["missing_fields"] = []
        if not isinstance(normalized["ambiguity_flags"], list):
            normalized["ambiguity_flags"] = []

        for field in REQUIRED_CLAIM_FIELDS:
            value = normalized.get(field)

            if field == "claim_amount":
                if value in (None, ""):
                    normalized["claim_amount"] = None
                    self._add_flag(normalized["missing_fields"], field)
                    continue
                try:
                    normalized["claim_amount"] = float(value)
                except (TypeError, ValueError):
                    normalized["claim_amount"] = None
                    self._add_flag(normalized["missing_fields"], field)
                    self._add_flag(normalized["ambiguity_flags"], field)
                continue

            cleaned = self._normalize_text_field(value)
            normalized[field] = cleaned
            if cleaned is None:
                self._add_flag(normalized["missing_fields"], field)
                continue

            if self._is_vague_placeholder(cleaned):
                self._add_flag(normalized["ambiguity_flags"], field)

        for flagged_field in tuple(normalized["ambiguity_flags"]):
            if flagged_field in {"diagnosis", "requested_service"}:
                field_value = normalized.get(flagged_field)
                if field_value is None or self._is_vague_placeholder(field_value):
                    normalized[flagged_field] = None
                    self._add_flag(normalized["missing_fields"], flagged_field)

        return normalized

    def _normalize_text_field(self, value):
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _is_vague_placeholder(self, value):
        return str(value).strip().lower() in VAGUE_TEXT_VALUES

    def _add_flag(self, items, value):
        if value not in items:
            items.append(value)
    
    def clinical_adjudication(self, diagnosis, requested_service, policy_context=None):
        """Perform AI-driven clinical adjudication for medical necessity
        
        This method evaluates whether a requested healthcare service is
        medically necessary for the given diagnosis, considering policy terms.
        
        Args:
            diagnosis (str): Medical diagnosis from the claim
            requested_service (str): Healthcare service being requested
            policy_context (dict): Policy terms and coverage details (optional)
            
        Returns:
            dict: Decision ('APPROVED'/'DENIED') with clinical reasoning
        """
        # Build policy context section if available
        context_section = ""
        if policy_context:
            context_section = f"\n\nPolicy Context:\n{json.dumps(policy_context, indent=2)}\n\nConsider the policy terms, coverage limits, and exclusions when making your decision."
        
        # Create clinical adjudication prompt with medical context
        prompt = Path("prompts/clinical_adjudication_prompt.txt")
        clinical_adjudication_prompt = prompt.read_text()\
                                        .replace("{diagnosis}", diagnosis or 'Not specified')\
                                        .replace("{requested_service}", requested_service or 'Not specified')\
                                        .replace("{context_section}", context_section)
        """You are a clinical adjudicator. Evaluate if the requested service is medically necessary for the diagnosis.{context_section}

Diagnosis: {diagnosis}
Requested Service: {requested_service}

Return ONLY a JSON object:
{{
  "decision": "APPROVED" or "DENIED",
  "reasoning": "brief clinical justification considering policy terms"
}}

Return JSON only."""

        # Call OpenAI API for clinical decision making
        response = openai.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a clinical adjudicator with access to policy information. Return only valid JSON."},
                {"role": "user", "content": clinical_adjudication_prompt}
            ],
            max_tokens=300
        )
        
        try:
            # Clean and parse JSON response
            content = response.choices[0].message.content.strip() # removes white spaces
            
            # Remove markdown formatting if present
            if content.startswith('```json'):
                content = content[7:-3].strip()
            elif content.startswith('```'):
                content = content[3:-3].strip()
                
            # Parse and return clinical decision
            return json.loads(content)
        except:
            # Return pending status if parsing fails
            return {"decision": "PENDING", "reasoning": "Unable to process"}
    
    def generate_claim_response_email(self, claim_data, decision, reasoning, claim_id, member_info=None):
        """Generate a professional email response for claim processing results
        
        This method creates personalized, empathetic emails that communicate
        claim decisions clearly while maintaining professional healthcare standards.
        
        Args:
            claim_data (dict): Original claim information
            decision (str): Claim decision ('APPROVED' or 'DENIED')
            reasoning (str): Explanation for the decision
            claim_id (str/int): Unique claim reference number
            member_info (dict): Member details for personalization (optional)
            
        Returns:
            str: Professional email body content
        """
        # Standardize decision status for consistent messaging
        status = "APPROVED" if decision == "APPROVED" else "DENIED"
        
        # Extract member name for personalization, with fallback
        if member_info:
            member_name = member_info.get('full_name', 'Valued Member')
        else:
            member_name = 'Valued Member'
        
        # Create comprehensive prompt for professional email generation
        prompt = Path("prompts/generate_claim_response_email.txt")
        prompt_content = prompt.read_text()
        response_email_prompt = prompt_content.replace("{claim_id}", str(claim_id))\
                                .replace("{member_name}", member_name)\
                                .replace("{member_id}", claim_data.get('member_id') or 'N/A')\
                                .replace("{diagnosis}", claim_data.get('diagnosis') or 'Not specified')\
                                .replace("{requested_service}", claim_data.get('requested_service') or 'N/A')\
                                .replace("{claim_amount}", str(claim_data.get('claim_amount') or 0))\
                                .replace("{status}", status)\
                                .replace("{reasoning}", reasoning)
        """Generate a professional healthcare claim response email with the following details:

Claim Reference: {claim_id}
Member: {member_name}
Member ID: {claim_data.get('member_id')}
Diagnosis: {claim_data.get('diagnosis')}
Requested Service: {claim_data.get('requested_service')}
Claim Amount: Ksh{claim_data.get('claim_amount')}
Decision: {status}
Reasoning: {reasoning}

The email should:
- Be professional and empathetic
- Include the claim reference number
- Clearly state the decision
- Provide the reasoning
- Include next steps if denied
- Be concise but complete

the email response should be a maximum of 50 words.
 
Return only the email body text, no subject line."""

        # Generate professional response email using GPT
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional healthcare claims processor writing response emails."},
                {"role": "user", "content": response_email_prompt}
            ],
            max_tokens=800
        )
        
        return response.choices[0].message.content
