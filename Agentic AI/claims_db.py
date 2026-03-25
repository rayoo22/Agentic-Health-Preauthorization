# Import necessary libraries for PostgreSQL database operations
import psycopg2
import os
from pathlib import Path

class ClaimsDB:
    """Database manager for healthcare claims processing and tracking
    
    This class handles all database operations related to claims including:
    - Creating new claim records
    - Updating claim statuses and adjudication results
    - Managing claim lifecycle from submission to final decision
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
    
    def insert_claim(self, member_id, diagnosis, requested_service, claim_amount):
        """Create a new claim record in the database
        
        Args:
            member_id (str): Unique identifier for the member
            diagnosis (str): Medical diagnosis from the claim
            requested_service (str): Healthcare service being requested
            claim_amount (float): Monetary amount of the claim
            
        Returns:
            int: Unique claim ID for the newly created claim
        """
        # Use cursor context manager for automatic resource cleanup
        with self.conn.cursor() as cur:
            # Insert new claim with PENDING status as default using parameterized query
            cur.execute("""
                INSERT INTO claims (member_id, diagnosis, requested_service, claim_amount, status)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING claim_id
            """, (member_id, diagnosis, requested_service, claim_amount, 'PENDING'))
            
            # Commit transaction to persist changes
            self.conn.commit()
            
            # Return the auto-generated claim ID
            return cur.fetchone()[0]
    
    def update_claim_status(self, claim_id, status, reasoning=None):
        """Update the status and adjudication reasoning for a claim
        
        Args:
            claim_id (int): Unique claim identifier
            status (str): New claim status ('APPROVED', 'DENIED', 'PENDING')
            reasoning (str, optional): Explanation for the decision
        """
        # Use cursor context manager for safe database operations
        with self.conn.cursor() as cur:
            # Update with reasoning if provided
            if reasoning:
                cur.execute("""
                    UPDATE claims 
                    SET status = %s, adjudication_reasoning = %s
                    WHERE claim_id = %s
                """, (status, reasoning, claim_id))
            else:
                # Update status only if no reasoning provided
                cur.execute("""
                    UPDATE claims 
                    SET status = %s
                    WHERE claim_id = %s
                """, (status, claim_id))
            
            # Commit changes to database
            self.conn.commit()
    
    def close(self):
        """Close the database connection to free resources"""
        self.conn.close()
