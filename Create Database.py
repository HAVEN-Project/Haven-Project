import sqlite3

#connect to (or create the database if missing)
conn = sqlite3.connect('StockData.db')

#create a cursor object
cursor = conn.cursor()

#create a table
cursor.execute('''
    CREATE TABLE StockData (
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL
    )
    
''')

cursor.execute('''
    CREATE INDEX idx_date_symbol ON StockData(date, symbol)
''')