#------------------------------------------------------------------------------
###            Common Libraries that are required to interact withSQL LITE DB
#------------------------------------------------------------------------------
import pandas as pd

import sqlite3
from loguru import logger
import warnings
import sys, os
import pretty_errors
# Import Configurations
from src.config import cfg_nifty as cfg_nifty
from src.config import cfg_vars as cfg_vars

warnings.filterwarnings("ignore", category=RuntimeWarning)


class SQLLiteManager:
    def __init__(self,db_path,table_name):
        self.db_path = db_path
        self.table_name = table_name
        logger.info(f"Initializing SQLLiteLoader for DB: {db_path}, Table: {table_name}")
        self.conn = sqlite3.connect(self.db_path)
        #self.out_table_name = out_table_name
        logger.info(f"                            ")
        logger.trace(f"SQLLiteLoader initialized for DB: {db_path}, Table: {table_name}")

    def read_data(self, query: str = None) -> pd.DataFrame:
        try:
            conn = None
            # Connect to the SQLite database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if query:
                #logger.info(f"Reading data using custom query: {query}")
                return pd.read_sql_query(query, conn)
            else:
                #logger.info(f"Reading all data from table: {self.table_name}")
                return pd.read_sql_query(f"SELECT * FROM {self.table_name}", conn)
        except Exception as e:
            logger.exception(f"Failed to read data: {e}")
            raise

    def write_data(self, df, truncate_and_load_flag):
        """
        Write DataFrame records to SQLite table.
        Creates the table if it doesn't exist.
        
        Args:
            df: pandas DataFrame to write
            truncate_and_load_flag: "yes" to truncate before loading, "no" to append
        
        Returns:
            bool: True if successful, False otherwise
        """
        conn = None
        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute(f"""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='{self.table_name}'
            """)
            table_exists = cursor.fetchone() is not None
            
            if truncate_and_load_flag == "yes" and table_exists:
                # Truncate by replacing the table
                logger.info(f"Truncating and replacing table {self.table_name}...")
                if_exists_mode = "replace"
            elif not table_exists:
                # Create new table
                logger.info(f"Creating new table {self.table_name}...")
                if_exists_mode = "fail"
            else:
                # Append to existing table
                logger.info(f"Appending to existing table {self.table_name}...")
                if_exists_mode = "append"
            
            # Write data to the table
            df.to_sql(self.table_name, conn, if_exists=if_exists_mode, index=False)
            
            # Commit the transaction
            conn.commit()
            
            logger.info(f"Successfully wrote {len(df)} rows to table {self.table_name}.")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error writing data to table {self.table_name}: {e}")
            if conn:
                conn.rollback()
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error writing data to table {self.table_name}: {e}")
            if conn:
                conn.rollback()
            return False
            
        finally:
            # Close connection
            if conn:
                conn.close()
    
    def close(self):
        self.conn.close()
        logger.info(f"Closed connection to database: {self.db_name}")

#db_path = cfg_vars.db_dir + cfg_vars.db_name
#loader= SQLLiteManager(db_path=db_path,table_name='historical_stocks')