# Import necessary libraries for PostgreSQL database operations
import psycopg2  # PostgreSQL database adapter for Python
from psycopg2.extras import execute_values, Json  # PostgreSQL extras for batch operations and JSON handling
import os  # Operating system interface for environment variables

class EmailDB:
    """Database manager for email storage and claim processing workflow
    
    This class handles all database operations related to:
    - Email metadata storage and duplicate prevention
    - Claim record management and status tracking
    - Database schema creation and maintenance
    """
    
    def __init__(self):
        """Initialize database connection and create required tables
        
        Connects to PostgreSQL database using credentials from .env file
        and ensures all required tables exist.
        """
        # Establish PostgreSQL connection using environment variables
        self.conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        
        # Create database tables if they don't exist
        self.create_table()
    
    def create_table(self):
        """Create required database tables for email and claim processing
        
        Creates two main tables:
        1. emails: Stores email metadata and processing status
        2. claims: Stores claim information and adjudication results
        """
        # Use cursor context manager for safe database operations
        with self.conn.cursor() as cur:
            # Create emails table for storing processed email metadata
            cur.execute("""
                CREATE TABLE IF NOT EXISTS emails (
                    message_id VARCHAR(255) PRIMARY KEY,     -- Gmail message ID (unique)
                    sender VARCHAR(255),                     -- Email sender address
                    subject TEXT,                            -- Email subject line
                    date TIMESTAMP,                          -- Email date/time
                    body_snippet TEXT,                       -- Email body preview
                    attachments JSONB,                       -- Email attachments (JSON format)
                    status VARCHAR(20) DEFAULT 'new',       -- Processing status
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Record creation time
                )
            """)
            
            # Create claims table for storing claim information and decisions
            cur.execute("""
                CREATE TABLE IF NOT EXISTS claims (
                    claim_id SERIAL PRIMARY KEY,            -- Auto-incrementing claim ID
                    member_id VARCHAR(100),                  -- Member identifier
                    diagnosis TEXT,                          -- Medical diagnosis
                    requested_service TEXT,                  -- Requested healthcare service
                    claim_amount DECIMAL(10, 2),            -- Claim monetary amount
                    adjudication_reasoning TEXT,             -- AI decision reasoning
                    status VARCHAR(20) DEFAULT 'NEW',       -- Claim processing status
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Record creation time
                )
            """)
            
            # Commit table creation changes
            self.conn.commit()
    
    def email_exists(self, message_id):
        """Check if an email has already been processed to prevent duplicates
        
        Args:
            message_id (str): Gmail message ID to check
            
        Returns:
            bool: True if email exists in database, False otherwise
        """
        # Use cursor context manager for safe database operations
        with self.conn.cursor() as cur:
            # Query for existing email with the given message ID
            cur.execute("SELECT 1 FROM emails WHERE message_id = %s", (message_id,))
            
            # Return True if any row found, False otherwise
            return cur.fetchone() is not None
    
    def insert_email(self, email_data):
        """Insert new email record if it doesn't already exist
        
        Args:
            email_data (dict): Email information including message_id, sender, subject, etc.
            
        Returns:
            bool: True if email was inserted (new), False if already exists (duplicate)
        """
        # Check for duplicates before insertion
        if self.email_exists(email_data['message_id']):
            return False
        
        # Use cursor context manager for safe database operations
        with self.conn.cursor() as cur:
            # Insert new email record with all metadata
            cur.execute("""
                INSERT INTO emails (message_id, sender, subject, date, body_snippet, attachments, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                email_data['message_id'],                    # Gmail message ID
                email_data['sender'],                        # Sender email address
                email_data['subject'],                       # Email subject
                email_data['date'],                          # Email timestamp
                email_data['body_snippet'],                  # Email preview text
                Json(email_data.get('attachments', [])),     # Attachments as JSON
                email_data.get('status', 'new')             # Processing status
            ))
            
            # Commit insertion
            self.conn.commit()
            return True
    
    def update_status(self, message_id, status):
        """Update the processing status of an email
        
        Args:
            message_id (str): Gmail message ID to update
            status (str): New status ('processed', 'failed', etc.)
        """
        # Use cursor context manager for safe database operations
        with self.conn.cursor() as cur:
            # Update email status
            cur.execute("UPDATE emails SET status = %s WHERE message_id = %s", (status, message_id))
            
            # Commit status update
            self.conn.commit()
    
    def insert_claim(self, claim_data):
        """Insert a new claim record into the database
        
        Args:
            claim_data (dict): Claim information including member_id, diagnosis, etc.
            
        Returns:
            int: Auto-generated claim ID for the new claim
        """
        # Use cursor context manager for safe database operations
        with self.conn.cursor() as cur:
            # Insert new claim with NEW status and return generated claim_id
            cur.execute("""
                INSERT INTO claims (member_id, diagnosis, requested_service, claim_amount, status)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING claim_id
            """, (
                claim_data.get('member_id'),         # Member identifier
                claim_data.get('diagnosis'),         # Medical diagnosis
                claim_data.get('requested_service'), # Requested service
                claim_data.get('claim_amount'),      # Claim amount
                'NEW'                                # Initial status
            ))
            
            # Commit insertion
            self.conn.commit()
            
            # Return the auto-generated claim ID
            return cur.fetchone()[0]
    
    def update_claim_status(self, claim_id, status):
        """Update the processing status of a claim
        
        Args:
            claim_id (int): Unique claim identifier
            status (str): New claim status ('APPROVED', 'DENIED', 'PENDING')
        """
        # Use cursor context manager for safe database operations
        with self.conn.cursor() as cur:
            # Update claim status
            cur.execute("UPDATE claims SET status = %s WHERE claim_id = %s", (status, claim_id))
            
            # Commit status update
            self.conn.commit()
    
    def get_claim(self, claim_id):
        """Retrieve claim information by claim ID
        
        Args:
            claim_id (int): Unique claim identifier
            
        Returns:
            dict: Claim information or None if not found
        """
        # Use cursor context manager for safe database operations
        with self.conn.cursor() as cur:
            # Query claim information
            cur.execute("SELECT member_id, diagnosis, requested_service, claim_amount, status FROM claims WHERE claim_id = %s", (claim_id,))
            
            # Fetch claim data
            row = cur.fetchone()
            
            # Return structured claim data if found
            if row:
                return {
                    'member_id': row[0],           # Member identifier
                    'diagnosis': row[1],           # Medical diagnosis
                    'requested_service': row[2],   # Requested service
                    'claim_amount': row[3],        # Claim amount
                    'status': row[4]               # Current status
                }
            
            # Return None if claim not found
            return None
    
    def close(self):
        """Close the database connection to free resources"""
        self.conn.close()
