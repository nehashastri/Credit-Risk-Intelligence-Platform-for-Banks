"""
Google Cloud Platform BigQuery Connection Module
Handles authentication and query execution for BigQuery operations
"""

import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import json
from typing import List, Dict, Any, Optional

class BigQueryConnector:
    """Handles BigQuery connection and query execution"""
    
    def __init__(self):
        """Initialize BigQuery client with credentials from Streamlit secrets"""
        try:
            # Get credentials from Streamlit secrets
            credentials_info = st.secrets["gcp_service_account"]
            
            # Create credentials object
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            
            # Initialize BigQuery client
            self.client = bigquery.Client(
                credentials=credentials,
                project=st.secrets["gcp"]["project_id"]
            )
            
            self.project_id = st.secrets["gcp"]["project_id"]
            self.dataset_id = st.secrets["gcp"]["dataset_id"]
            
        except Exception as e:
            st.error(f"Failed to initialize BigQuery client: {str(e)}")
            raise e
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a BigQuery SQL query and return results as list of dictionaries
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            List[Dict[str, Any]]: Query results as list of dictionaries
        """
        try:
            query_job = self.client.query(query)
            results = query_job.result()
            
            # Convert to list of dictionaries
            return [dict(row) for row in results]
            
        except Exception as e:
            st.error(f"Query execution failed: {str(e)}")
            raise e
    
    def execute_query_to_dataframe(self, query: str) -> pd.DataFrame:
        """
        Execute a BigQuery SQL query and return results as pandas DataFrame
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            pd.DataFrame: Query results as DataFrame
        """
        try:
            return self.client.query(query).to_dataframe()
        except Exception as e:
            st.error(f"Query execution failed: {str(e)}")
            raise e
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get schema information for a specific table
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            List[Dict[str, Any]]: Table schema information
        """
        try:
            table_ref = self.client.dataset(self.dataset_id).table(table_name)
            table = self.client.get_table(table_ref)
            
            schema_info = []
            for field in table.schema:
                schema_info.append({
                    'name': field.name,
                    'type': field.field_type,
                    'mode': field.mode,
                    'description': field.description
                })
            
            return schema_info
            
        except Exception as e:
            st.error(f"Failed to get table schema for {table_name}: {str(e)}")
            return []
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get basic information about a table (row count, size, etc.)
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            Dict[str, Any]: Table information
        """
        try:
            # Get row count
            count_query = f"""
            SELECT COUNT(*) as row_count
            FROM `{self.project_id}.{self.dataset_id}.{table_name}`
            """
            
            count_result = self.execute_query(count_query)
            row_count = count_result[0]['row_count'] if count_result else 0
            
            # Get table size
            size_query = f"""
            SELECT 
                ROUND(SUM(size_bytes) / 1024 / 1024, 2) as size_mb
            FROM `{self.project_id}.{self.dataset_id}.__TABLES__`
            WHERE table_id = '{table_name}'
            """
            
            size_result = self.execute_query(size_query)
            size_mb = size_result[0]['size_mb'] if size_result else 0
            
            # Get latest timestamp - use table-specific columns directly
            latest_timestamp = None
            
            # Define timestamp columns to try based on table name
            if table_name == 'news_articles':
                timestamp_columns = ['ingest_date', 'ingest_datetime', 'published_at']
            elif table_name in ['fact_credit_outcomes', 'fact_macro_indicators_daily', 'fact_macro_indicators_monthly']:
                timestamp_columns = ['date']
            else:
                timestamp_columns = ['date']  # Only try 'date' as fallback
            
            for col in timestamp_columns:
                try:
                    timestamp_query = f"""
                    SELECT MAX({col}) as latest_timestamp
                    FROM `{self.project_id}.{self.dataset_id}.{table_name}`
                    """
                    timestamp_result = self.execute_query(timestamp_query)
                    if timestamp_result and timestamp_result[0]['latest_timestamp']:
                        latest_timestamp = timestamp_result[0]['latest_timestamp']
                        break
                except Exception as e:
                    continue
            
            return {
                'row_count': row_count,
                'size_mb': size_mb,
                'latest_timestamp': latest_timestamp,
                'table_name': table_name
            }
            
        except Exception as e:
            st.error(f"Failed to get table info for {table_name}: {str(e)}")
            return {
                'row_count': 0,
                'size_mb': 0,
                'latest_timestamp': None,
                'table_name': table_name,
                'error': str(e)
            }
    
    def get_missing_data_summary(self, table_name: str) -> Dict[str, Any]:
        """
        Get summary of missing data for a table
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            Dict[str, Any]: Missing data summary
        """
        try:
            # Get schema to identify columns
            schema = self.get_table_schema(table_name)
            columns = [field['name'] for field in schema]
            
            missing_data = {}
            
            for column in columns:
                # Skip timestamp/date columns based on table type
                if table_name == 'news_articles':
                    skip_columns = ['ingest_date', 'ingest_datetime', 'published_at']
                else:
                    skip_columns = ['timestamp', 'date', 'created_at', 'updated_at']
                
                if column.lower() not in [col.lower() for col in skip_columns]:
                    query = f"""
                    SELECT 
                        COUNT(*) as total_rows,
                        COUNT({column}) as non_null_rows,
                        COUNT(*) - COUNT({column}) as null_rows
                    FROM `{self.project_id}.{self.dataset_id}.{table_name}`
                    """
                    
                    result = self.execute_query(query)
                    if result:
                        total = result[0]['total_rows']
                        non_null = result[0]['non_null_rows']
                        null_count = result[0]['null_rows']
                        
                        missing_data[column] = {
                            'total_rows': total,
                            'non_null_rows': non_null,
                            'null_rows': null_count,
                            'null_percentage': (null_count / total * 100) if total > 0 else 0
                        }
            
            return missing_data
            
        except Exception as e:
            st.error(f"Failed to get missing data summary for {table_name}: {str(e)}")
            return {}
    
    def test_connection(self) -> bool:
        """
        Test the BigQuery connection
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Simple query to test connection
            test_query = "SELECT 1 as test"
            result = self.execute_query(test_query)
            return len(result) > 0 and result[0]['test'] == 1
        except Exception as e:
            st.error(f"Connection test failed: {str(e)}")
            return False
