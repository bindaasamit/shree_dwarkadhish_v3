import sqlite3
"""
# Connect to a database file (creates it if it doesn't exist)
conn = sqlite3.connect('data/db/shree_dwarkadhish.db')
# Create a cursor object
cursor_obj = conn.cursor()

# Execute the CREATE TABLE command
table_creation_query = """
"""
CREATE TABLE IF NOT EXISTS Forex(
    Date_GMT  TIMESTAMP NOT NULL,
    Open REAL,
    High REAL,
    Low	  REAL,
    Close  REAL,
    Volume REAL
);
"""
"""
cursor_obj.execute(table_creation_query)
print("Table created successfully")

# Commit the changes and close the connection
conn.commit()
conn.close()
"""

file_path = "C:/Users/Amit/Downloads/Abhishek/2019_2026_GBPJPY_M15.csv"
#read a file and insert data into the table
def insert_forex_data(file_path):
    import pandas as pd
    import sqlite3

    # Read the data from Excel file
    forex_df = pd.read_csv(file_path)

    # Connect to the database
    conn = sqlite3.connect('data/db/shree_dwarkadhish.db')
    cursor_obj = conn.cursor()

    # Insert data into the Forex table
    forex_df.to_sql('Forex', conn, if_exists='append', index=False)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

insert_forex_data(file_path)