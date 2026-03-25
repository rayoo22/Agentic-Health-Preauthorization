import psycopg2 
import os 
from pathlib import Path 

class MembersDB:
    """Database manager for healthcare member information and policy balances
    
    This class handles all database operations related to members including:
    - Retrieving member information and policy details
    - Managing policy balances and deductions
    - Validating member existence in the system
    """
    
    def __init__(self):
        """Initialize database connection using environment variables
        
        Connects to PostgreSQL database using credentials from .env file:
        - DB_HOST: Database server hostname
        - DB_PORT: Database server port
        - DB_NAME: Database name
        - DB_USER: Database username
        - DB_PASSWORD: Database password
        """
        # Establish PostgreSQL connection using environment variables
        self.conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
    
    def get_member(self, member_id):
        """Retrieve complete member information by member ID
        
        Args:
            member_id (str): Unique identifier for the member
            
        Returns:
            dict: Member information including personal details and policy balance,
                  or None if member not found
        """
        # Use cursor context manager for safe database operations
        with self.conn.cursor() as cur:
            # Query member table for complete member information
            cur.execute("""
                SELECT member_id, full_name, date_of_birth, policy_id, status, policy_balance 
                FROM members 
                WHERE member_id = %s
            """, (member_id,))
            
            # Fetch the first (and should be only) matching row
            row = cur.fetchone()
            
            # If member found, return structured dictionary
            if row:
                return {
                    'member_id': row[0],        # Unique member identifier
                    'full_name': row[1],        # Member's full name
                    'date_of_birth': row[2],    # Member's date of birth
                    'policy_id': row[3],        # Associated policy identifier
                    'status': row[4],           # Member status (active, inactive, etc.)
                    'policy_balance': float(row[5])  # Available policy balance
                }
            
            # Return None if member not found
            return None
    
    def deduct_from_balance(self, member_id, amount):
        """Deduct approved claim amount from member's policy balance
        
        This method is called when a claim is approved to reduce the
        member's available policy balance by the claim amount.
        
        Args:
            member_id (str): Unique identifier for the member
            amount (float): Amount to deduct from policy balance
        """
        # Use cursor context manager for safe database operations
        with self.conn.cursor() as cur:
            # Update member's policy balance by subtracting claim amount
            cur.execute("""
                UPDATE members 
                SET policy_balance = policy_balance - %s 
                WHERE member_id = %s
            """, (amount, member_id))
            
            # Commit transaction to persist balance change
            self.conn.commit()
    
    def close(self):
        """Close the database connection to free resources"""
        self.conn.close()
