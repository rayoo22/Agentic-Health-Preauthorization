# Import necessary libraries and modules for claims workflow processing
from dotenv import load_dotenv  # Load environment variables from .env file
from gmail_reader import GmailReader  # Gmail API integration for email handling
from email_extractor import EmailExtractor  # Email content extraction utilities
from openai_agent import OpenAIEmailAgent  # AI agent for claim processing
from db_manager import EmailDB  # Database management for email and claim storage

def process_claims_workflow():
    """Alternative claims processing workflow for testing and development
    
    This function provides a simplified workflow for processing claims that:
    1. Fetches a limited number of emails for testing
    2. Extracts claim data using AI
    3. Performs clinical adjudication
    4. Updates claim statuses through the workflow
    
    Note: This is a development/testing version of the main workflow
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize all required components
    gmail = GmailReader()  # Gmail API client
    extractor = EmailExtractor(gmail.service)  # Email content extractor
    agent = OpenAIEmailAgent()  # AI processing agent
    db = EmailDB()  # Database manager
    
    print("Fetching emails...")
    
    # Fetch a limited number of emails for processing (testing purposes)
    results = gmail.service.users().messages().list(userId='me', maxResults=10).execute()
    messages = results.get('messages', [])
    
    print(f"Processing {len(messages)} emails...\n")
    
    # Process each email through the simplified workflow
    for msg in messages:
        # STEP 1: Extract complete email content
        email_content = extractor.extract_full_content(msg['id'])
        print(f"📧 {email_content['subject'][:60]}")
        
        # STEP 2: Use AI to extract structured claim data
        claim_data = agent.extract_claim_data(email_content['subject'], email_content['body'])
        
        # Validate claim data extraction
        if not claim_data:
            print("   ⊘ No claim data found\n")
            continue
        
        # Display extracted claim information
        print(f"   Member: {claim_data.get('member_id')}")
        print(f"   Diagnosis: {claim_data.get('diagnosis')}")
        print(f"   Service: {claim_data.get('requested_service')}")
        print(f"   Amount: ${claim_data.get('claim_amount')}")
        
        # STEP 3: Store claim in database with initial NEW status
        try:
            claim_id = db.insert_claim(claim_data)
            print(f"   ✓ Stored as Claim ID: {claim_id} [Status: NEW]")
            
            # STEP 4: Update status to IN_PROGRESS for processing
            db.update_claim_status(claim_id, 'IN_PROGRESS')
            
            # STEP 5: Perform AI-driven clinical adjudication
            adjudication = agent.clinical_adjudication(
                claim_data['diagnosis'],
                claim_data['requested_service']
            )
            
            # Display adjudication results
            print(f"   Decision: {adjudication['decision']}")
            print(f"   Reasoning: {adjudication['reasoning']}")
            
            # STEP 6: Update final status to PROCESSED
            db.update_claim_status(claim_id, 'PROCESSED')
            print(f"   ✓ Updated to [Status: PROCESSED]\n")
            
        except Exception as e:
            # Handle any processing errors
            print(f"   ✗ Error: {e}")
            
            # Update claim status to FAILED if error occurred
            if 'claim_id' in locals():
                db.update_claim_status(claim_id, 'FAILED')
            print()
    
    # Clean up database connection
    db.close()
    print("Claims processing complete!")

# Entry point - execute workflow when script is run directly
if __name__ == "__main__":
    process_claims_workflow()
