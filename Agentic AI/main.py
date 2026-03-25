from dotenv import load_dotenv
from gmail_reader import GmailReader
from db_manager import EmailDB
from openai_agent import OpenAIEmailAgent
from claims_db import ClaimsDB
from members_db import MembersDB
from policies_db import PoliciesDB
from RAG.query_policy_rag import query_policy_rag, format_policy_context
from vertex_judge import VertexClaimJudge
import base64
import os


def format_judge_email_body(email, claim_id, claim_data, member, final_decision, final_reasoning, judge_result):
    member_name = member.get('full_name') if member else 'Unknown'
    judge_flags = judge_result.get('judge_flags', []) or []
    flags_text = "\n".join(f"- {flag}" for flag in judge_flags) if judge_flags else "- None"
    stage_evaluation = judge_result.get('stage_evaluation', {}) or {}

    def stage_text(stage_name):
        stage = stage_evaluation.get(stage_name, {}) or {}
        return f"Score: {stage.get('score', 'N/A')} | Assessment: {stage.get('assessment', 'N/A')}"

    return f"""Vertex AI Judge Evaluation Report

Original Email Subject: {email.get('subject', 'N/A')}
Original Sender: {email.get('sender', 'N/A')}
Claim Reference: {claim_id}

Extracted Claim Data
- Member ID: {claim_data.get('member_id', 'N/A')}
- Member Name: {member_name}
- Diagnosis: {claim_data.get('diagnosis', 'N/A')}
- Requested Service: {claim_data.get('requested_service', 'N/A')}
- Claim Amount: {claim_data.get('claim_amount', 'N/A')}

Primary Workflow Output
- Final Decision: {final_decision}
- Reasoning: {final_reasoning}

Vertex Judge Output
- Judge Agrees: {judge_result.get('judge_agrees')}
- Judge Verdict: {judge_result.get('judge_verdict')}
- Judge Score: {judge_result.get('judge_score')}
- Judge Reasoning: {judge_result.get('judge_reasoning')}

Stage Evaluation
- Extraction: {stage_text('extraction')}
- Member Validation: {stage_text('member_validation')}
- Balance Check: {stage_text('balance_check')}
- Policy Alignment: {stage_text('policy_alignment')}
- Final Decision: {stage_text('final_decision')}

Judge Flags
{flags_text}
"""


def maybe_send_judge_email(gmail, judge, email, claim_id, claim_data, member, policy_context, final_decision, final_reasoning):
    judge_recipient = os.getenv('JUDGE_REPORT_EMAIL')
    if not judge or not judge_recipient:
        return

    try:
        judge_result = judge.judge_claim_decision(
            email_subject=email['subject'],
            email_body=email.get('full_body', ''),
            claim_data=claim_data,
            member_info=member,
            policy_context=policy_context,
            proposed_decision=final_decision,
            proposed_reasoning=final_reasoning,
        )
    except Exception as e:
        print(f"    Judge evaluation failed: {e}")
        return

    judge_subject = f"Judge Evaluation - Claim #{claim_id} - {final_decision}"
    judge_body = format_judge_email_body(
        email,
        claim_id,
        claim_data,
        member,
        final_decision,
        final_reasoning,
        judge_result,
    )
    judge_message_id = gmail.send_reply(
        email['message_id'],
        judge_subject,
        judge_body,
        judge_recipient,
    )
    if judge_message_id:
        print(f"    Judge report email sent (ID: {judge_message_id})")
    else:
        print("    Failed to send judge report email")

def extract_email_body(gmail_service, msg_id):
    msg = gmail_service.users().messages().get(userId='me', id=msg_id, format='full').execute()
    payload = msg['payload']
    
    if 'body' in payload and payload['body'].get('data'):
        return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
    
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain' and part['body'].get('data'):
                return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
    
    return msg.get('snippet', '')

""" this is the main workflow executing all requests """
def main():
    load_dotenv()
    
    print("Fetching unread emails from Gmail (Agentic_AI label)...")
    gmail = GmailReader()
    emails = gmail.fetch_emails(max_results=os.getenv('MAX_EMAILS'), label_name='Agentic_AI', unread_only=True)
    print(f"Found {len(emails)} unread emails\n")
    
    """creating an instance of functions to be used"""
    email_db = EmailDB()
    claims_db = ClaimsDB()
    members_db = MembersDB()
    policies_db = PoliciesDB()
    agent = OpenAIEmailAgent()
    judge = VertexClaimJudge() if os.getenv('ENABLE_VERTEX_JUDGE', 'false').lower() == 'true' else None
    
    new_count = 0
    duplicate_count = 0
    
    """iterate through emails and process each one"""""
    for email in emails:
        if email_db.insert_email(email):
            new_count += 1
            print(f"Added email: {email['subject'][:50]}")
            
            # Extract full body
            full_body = extract_email_body(gmail.service, email['message_id'])
            email['full_body'] = full_body
            print(full_body)
            #input("Press Enter to continue...")
        
            # Extract claim data using LLM
            claim_data = agent.extract_claim_data(email['subject'], full_body)
            print(claim_data)
            #input("Press Enter to continue...")

            if not claim_data:
                print("  Could not extract claim data\n")
                continue
            
            print(f"  Extracted: Member {claim_data.get('member_id')}, Ksh. {claim_data.get('claim_amount')}")
            
            # Check member existence and policy balance
            member = members_db.get_member(claim_data['member_id'])
            
            """if member is not found status column will be labeled DENIED\n"""
            if not member:
                print(f"   Member {claim_data['member_id']} not found")
                claim_id = claims_db.insert_claim(
                    claim_data['member_id'],
                    claim_data['diagnosis'],
                    claim_data['requested_service'],
                    claim_data['claim_amount']
                )
                claims_db.update_claim_status(
                    claim_id,
                    'DENIED',
                    'Member not found in system'
                )
                
                # Send denial response email
                response_body = agent.generate_claim_response_email(
                    claim_data, 
                    'DENIED', 
                    'Member not found in system', 
                    claim_id
                )
                response_subject = f"Claim Decision - Reference #{claim_id} - DENIED"
                reply_id = gmail.send_reply(
                    email['message_id'],
                    response_subject,
                    response_body,
                    email['sender']
                )
                if reply_id:
                    print(f"     Denial response email sent (ID: {reply_id})")
                maybe_send_judge_email(
                    gmail,
                    judge,
                    email,
                    claim_id,
                    claim_data,
                    None,
                    "",
                    'DENIED',
                    'Member not found in system',
                )
                gmail.mark_as_read(email['message_id'])
                continue
            
            print(f"   Member found: {member['full_name']} (Balance: Ksh.{member['policy_balance']})")
            
            # Check policy balance
            if member['policy_balance'] <= claim_data['claim_amount']:
                print(f"  Insufficient balance: ${member['policy_balance']} < Ksh. {claim_data['claim_amount']}")
                claim_id = claims_db.insert_claim(
                    claim_data['member_id'],
                    claim_data['diagnosis'],
                    claim_data['requested_service'],
                    claim_data['claim_amount']
                )
                denial_reason = f'Insufficient policy balance: Ksh. {member["policy_balance"]} available, Ksh. {claim_data["claim_amount"]} required'
                claims_db.update_claim_status(
                    claim_id,
                    'DENIED',
                    denial_reason
                )
                
                # Send denial response email
                response_body = agent.generate_claim_response_email(
                    claim_data, 
                    'DENIED', 
                    denial_reason, 
                    claim_id, 
                    member
                )
                response_subject = f"Claim Decision - Reference #{claim_id} - DENIED"
                reply_id = gmail.send_reply(
                    email['message_id'],
                    response_subject,
                    response_body,
                    email['sender']
                )
                if reply_id:
                    print(f"     Denial response email sent (ID: {reply_id})")
                maybe_send_judge_email(
                    gmail,
                    judge,
                    email,
                    claim_id,
                    claim_data,
                    member,
                    "",
                    'DENIED',
                    denial_reason,
                )
                print(f"    Claim ID {claim_id}: DENIED - Insufficient balance\n")
                gmail.mark_as_read(email['message_id'])
                continue
            
            # Store in claims table
            claim_id = claims_db.insert_claim(
                claim_data['member_id'],
                claim_data['diagnosis'],
                claim_data['requested_service'],
                claim_data['claim_amount']
            )
            print(f"    Stored as Claim ID: {claim_id}")
            
            # Retrieve policy context using RAG
            relevant_chunks = query_policy_rag(
                claim_data['diagnosis'],
                claim_data['requested_service']
            )
            policy_context = format_policy_context(relevant_chunks)
            print(f"    Retrieved {len(relevant_chunks)} relevant policy sections via RAG")
            
            # Clinical adjudication using LLM with RAG (only if balance is sufficient)
            adjudication = agent.clinical_adjudication(
                claim_data['diagnosis'],
                claim_data['requested_service'],
                policy_context
            )
            
            # Final decision combines clinical and financial checks
            final_decision = adjudication['decision']
            final_reasoning = f"Policy balance: Ksh. {member['policy_balance']}. Clinical: {adjudication['reasoning']}"
            
            # Update claim status
            claims_db.update_claim_status(
                claim_id,
                final_decision,
                final_reasoning
            )
            
            # Generate and send response email
            response_body = agent.generate_claim_response_email(
                claim_data, 
                final_decision, 
                final_reasoning, 
                claim_id, 
                member
            )
            
            response_subject = f"Claim Decision - Reference #{claim_id} - {final_decision}"
            
            # Send reply email
            reply_id = gmail.send_reply(
                email['message_id'],
                response_subject,
                response_body,
                email['sender']
            )
            
            if reply_id:
                print(f"    Response email sent (ID: {reply_id})")
            else:
                print(f"    Failed to send response email")

            maybe_send_judge_email(
                gmail,
                judge,
                email,
                claim_id,
                claim_data,
                member,
                policy_context,
                final_decision,
                final_reasoning,
            )
            
            # Deduct from policy balance if approved
            if final_decision == 'APPROVED':
                members_db.deduct_from_balance(claim_data['member_id'], claim_data['claim_amount'])
                new_balance = member['policy_balance'] - claim_data['claim_amount']
                print(f"    Adjudication: {final_decision}")
                print(f"    Reasoning: {final_reasoning}")
                print(f"    Deducted Ksh {claim_data['claim_amount']} from policy balance")
                print(f"    New balance: Ksh {new_balance}\n")
            else:
                print(f"    Adjudication: {final_decision}")
                print(f"    Reasoning: {final_reasoning}\n")
            
            # Mark email as read in Gmail
            gmail.mark_as_read(email['message_id'])
            print(f"    Marked as read in Gmail")
        else:
            duplicate_count += 1
            print(f"Skipped (duplicate): {email['subject'][:50]}")
    
    email_db.close()
    claims_db.close()
    members_db.close()
    policies_db.close()
    
    print(f"\nSummary: {new_count} new emails processed, {duplicate_count} duplicates skipped")

if __name__ == "__main__":
    main()
