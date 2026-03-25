import psycopg2
import os
from decimal import Decimal
from datetime import date

class PoliciesDB:
    """Database manager for healthcare policy information and coverage details
    
    This class handles all database operations related to policies including:
    - Retrieving policy terms and coverage details
    - Providing policy context for clinical adjudication (RAG)
    - Managing policy-specific rules and exclusions
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
    
    def get_policy(self, policy_id):
        """Retrieve complete policy information by policy ID
        
        This method fetches all policy details including coverage terms,
        exclusions, limits, and other relevant information needed for
        AI-driven clinical adjudication (RAG - Retrieval Augmented Generation).
        
        Args:
            policy_id (str): Unique identifier for the policy
            
        Returns:
            dict: Complete policy information with all fields,
                  or None if policy not found
        """
        # Use cursor context manager for safe database operations
        with self.conn.cursor() as cur:
            # Query policies table for complete policy information
            cur.execute("SELECT * FROM policies WHERE policy_id = %s", (policy_id,))
            
            # Get column names from cursor description for dynamic dict creation
            columns = [desc[0] for desc in cur.description]
            
            # Fetch the first (and should be only) matching row
            row = cur.fetchone()
            
            # If policy found, create structured dictionary
            if row:
                # Create dictionary mapping column names to values
                policy_dict = dict(zip(columns, row))
                
                # Convert Decimal and date types for JSON serialization compatibility
                # This is important for AI model consumption and API responses
                for key, value in policy_dict.items():
                    if isinstance(value, Decimal):
                        policy_dict[key] = float(value)
                    elif isinstance(value, date):
                        policy_dict[key] = value.isoformat()
                
                return policy_dict
            
            # Return None if policy not found
            return None
    
    def close(self):
        """Close the database connection to free resources"""
        self.conn.close()
