# Import necessary libraries for Gmail API integration
from google.oauth2.credentials import Credentials 
from google_auth_oauthlib.flow import InstalledAppFlow 
from google.auth.transport.requests import Request 
from googleapiclient.discovery import build 
import os  
import pickle
from datetime import datetime 
from email.utils import parsedate_to_datetime  
import base64  
from email.mime.text import MIMEText  

# Gmail API scope - allows reading, modifying, and sending emails
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

class GmailReader:
    """Gmail API client for reading emails and sending responses in healthcare claims processing"""
    
    def __init__(self):
        """Initialize Gmail service with authentication"""
        # Authenticate and build Gmail service object
        self.service = self._authenticate()
    
    def _authenticate(self):
        """Handle OAuth2 authentication flow for Gmail API access"""
        creds = None
        
        # Check if we have stored credentials from previous authentication
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials exist, initiate authentication flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    # Refresh expired credentials using refresh token
                    creds.refresh(Request())
                except Exception as e:
                    # If refresh fails, delete token and re-authenticate
                    print(f"Token refresh failed: {e}")
                    print("Deleting expired token and re-authenticating...")
                    if os.path.exists('token.pickle'):
                        os.remove('token.pickle')
                    creds = None
            
            # Start new OAuth2 flow if no valid credentials
            if not creds:
                # Start new OAuth2 flow using credentials.json
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for future use to avoid re-authentication
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        # Build and return Gmail service object
        return build('gmail', 'v1', credentials=creds)
    
    def fetch_emails(self, max_results=10, label_name=None, unread_only=False):
        """Fetch emails from Gmail based on specified criteria
        
        Args:
            max_results (int): Maximum number of emails to retrieve
            label_name (str): Gmail label to filter emails (e.g., 'Agentic_AI')
            unread_only (bool): If True, only fetch unread emails
            
        Returns:
            list: List of email dictionaries with metadata
        """
        # Build Gmail search query based on parameters
        query_parts = []
        
        # Add label filter or default to inbox
        if label_name:
            query_parts.append(f'label:{label_name}')
        else:
            query_parts.append('in:inbox')
        
        # Add unread filter if requested
        if unread_only:
            query_parts.append('is:unread')
        
        # Combine query parts into single search string
        query = ' '.join(query_parts)
        
        # Execute Gmail API call to list messages matching criteria
        results = self.service.users().messages().list(userId='me', q=query, maxResults=max_results).execute()
        messages = results.get('messages', [])
        
        # Extract detailed information for each message
        emails = []
        for msg in messages:
            email_data = self._get_email_details(msg['id'])
            if email_data:
                emails.append(email_data)
        
        return emails
    
    def _get_email_details(self, msg_id):
        """Extract detailed information from a specific email message
        
        Args:
            msg_id (str): Gmail message ID
            
        Returns:
            dict: Email metadata including sender, subject, date, and snippet
        """
        # Fetch full message details from Gmail API
        msg = self.service.users().messages().get(userId='me', id=msg_id, format='full').execute()
        
        # Extract headers from email payload
        headers = msg['payload']['headers']
        
        # Parse important header fields with fallback values
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
        date_str = next((h['value'] for h in headers if h['name'] == 'Date'), None)
        
        # Convert date string to datetime object
        date = parsedate_to_datetime(date_str) if date_str else None
        
        # Return structured email data
        return {
            'message_id': msg['id'],
            'sender': sender,
            'subject': subject,
            'date': date,
            'body_snippet': msg.get('snippet', '')
        }
    
    def mark_as_read(self, msg_id):
        """Mark an email as read by removing the UNREAD label
        
        Args:
            msg_id (str): Gmail message ID to mark as read
        """
        # Remove UNREAD label from the message
        self.service.users().messages().modify(
            userId='me',
            id=msg_id,
            body={'removeLabelIds': ['UNREAD']}
        ).execute()
    
    def send_reply(self, original_msg_id, reply_subject, reply_body, recipient_email):
        """Send a reply email with claim processing results
        
        Args:
            original_msg_id (str): ID of original message to reply to
            reply_subject (str): Subject line for the reply
            reply_body (str): Body content of the reply
            recipient_email (str): Email address to send reply to
            
        Returns:
            str: Message ID of sent reply, or None if failed
        """
        # Create MIME text message with reply content
        message = MIMEText(reply_body)
        message['to'] = recipient_email
        message['subject'] = reply_subject
        
        # Encode message as base64 for Gmail API
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        # Prepare message payload with thread ID to maintain conversation
        send_message = {
            'raw': raw_message,
            'threadId': self._get_thread_id(original_msg_id)
        }
        
        try:
            # Send the reply through Gmail API
            result = self.service.users().messages().send(userId='me', body=send_message).execute()
            return result['id']
        except Exception as e:
            print(f"Error sending reply: {e}")
            return None

    def send_email(self, subject, body, recipient_email):
        """Send a standalone email message.
        
        Args:
            subject (str): Subject line for the email
            body (str): Body content of the email
            recipient_email (str): Email address to send to
            
        Returns:
            str: Message ID of sent email, or None if failed
        """
        message = MIMEText(body)
        message['to'] = recipient_email
        message['subject'] = subject

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        try:
            result = self.service.users().messages().send(
                userId='me',
                body={'raw': raw_message}
            ).execute()
            return result['id']
        except Exception as e:
            print(f"Error sending email: {e}")
            return None
    
    def _get_thread_id(self, msg_id):
        """Get thread ID for the original message to maintain conversation thread
        
        Args:
            msg_id (str): Gmail message ID
            
        Returns:
            str: Thread ID of the message, or None if not found
        """
        try:
            # Fetch message to get its thread ID
            msg = self.service.users().messages().get(userId='me', id=msg_id).execute()
            return msg['threadId']
        except:
            return None
