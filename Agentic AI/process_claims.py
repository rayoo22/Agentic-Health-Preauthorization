from dotenv import load_dotenv 
from gmail_reader import GmailReader
from openai_agent import OpenAIEmailAgent 
from claims_db import ClaimsDB 
import base64  

def extract_email_body(gmail_service, msg_id):
    """Extract the full body content from a Gmail message
    
    This function handles different email formats and structures to extract
    the complete text content from Gmail messages for AI processing.
    
    Args:
        gmail_service: Authenticated Gmail API service object
        msg_id (str): Gmail message ID to extract content from
        
    Returns:
        str: Decoded email body content or snippet if body unavailable
    """
    # Fetch the complete message with full format from Gmail API
    msg = gmail_service.users().messages().get(userId='me', id=msg_id, format='full').execute()
    payload = msg['payload']
    
    # Check if body data is directly available in the payload
    if 'body' in payload and payload['body'].get('data'):
        # Decode base64 encoded body content
        return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
    
    # If body is in parts (multipart message), search for text/plain content
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain' and part['body'].get('data'):
                # Decode base64 encoded part content
                return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
    
    # Fallback to message snippet if full body unavailable
    return msg.get('snippet', '')

def process_claims():
    """Simplified claims processing workflow for basic email-to-claim conversion
    
    This function provides a streamlined workflow that:
    1. Fetches emails from Gmail
    2. Extracts claim data using AI
    3. Stores claims in database
    4. Performs clinical adjudication
    5. Updates claim statuses
    
    Note: This is a simplified version without member validation or email responses
    """
    # Load environment variables from .env file (API keys, database URLs, etc.)
    load_dotenv()
    
    print("Fetching emails from Gmail...")
    
    # Initialize Gmail reader and fetch emails
    gmail = GmailReader()
    emails = gmail.fetch_emails(max_results=10)  # Limit to 10 emails for processing
    print(f"Found {len(emails)} emails\n")
    
    # Initialize AI agent and database connection
    agent = OpenAIEmailAgent()  # AI agent for data extraction and adjudication
    claims_db = ClaimsDB()  # Claims database manager
    
    # Process each email through the simplified workflow
    for email in emails:
        print(f"Processing: {email['subject']}")
        
        # STEP 1: Extract complete email body content
        full_body = extract_email_body(gmail.service, email['message_id'])
        
        # STEP 2: Use AI to extract structured claim data from email content
        claim_data = agent.extract_claim_data(email['subject'], full_body)
        
        # Validate that claim data was successfully extracted
        if not claim_data:
            print("  ✗ Could not extract claim data\n")
            continue
        
        # Display extracted claim information
        print(f"  Extracted: Member {claim_data.get('member_id')}, ${claim_data.get('claim_amount')}")
        
        # STEP 3: Store claim in database with initial PENDING status
        claim_id = claims_db.insert_claim(
            claim_data['member_id'],        # Member identifier
            claim_data['diagnosis'],        # Medical diagnosis
            claim_data['requested_service'], # Requested healthcare service
            claim_data['claim_amount']      # Claim monetary amount
        )
        print(f"  ✓ Stored as Claim ID: {claim_id}")
        
        # STEP 4: Perform AI-driven clinical adjudication
        # This evaluates medical necessity without policy context
        adjudication = agent.clinical_adjudication(
            claim_data['diagnosis'],
            claim_data['requested_service']
        )
        
        # STEP 5: Update claim status with adjudication results
        claims_db.update_claim_status(
            claim_id,                      # Claim identifier
            adjudication['decision'],      # APPROVED or DENIED
            adjudication['reasoning']      # Clinical reasoning
        )
        
        # Display adjudication results
        print(f"  ✓ Adjudication: {adjudication['decision']}")
        print(f"    Reasoning: {adjudication['reasoning']}\n")
    
    # Clean up database connection
    claims_db.close()
    print("Claims processing complete!")

# Entry point - execute claims processing when script is run directly
if __name__ == "__main__":
    process_claims()
