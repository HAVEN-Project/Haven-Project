import sqlite3

#connect to (or create the database if missing)
conn = sqlite3.connect('StockData.db')

#create a cursor object
cursor = conn.cursor()

#create a table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS HourlyData (
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        UNIQUE (symbol, date)
    )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS DailyData (
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        UNIQUE (symbol, date)
    )
''')
cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_hourly_symbol ON HourlyData(symbol)
''')
cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_daily_symbol ON DailyData(symbol)
''')
cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_Hourly_symbol_date ON HourlyData(symbol, date)
''')
cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_daily_symbol_date ON DailyData(symbol, date)
''')
