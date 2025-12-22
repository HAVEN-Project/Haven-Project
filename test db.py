import sqlite3

#connect to (or create the database if missing)
conn = sqlite3.connect('TestStockData.db')

#create a cursor object
cursor = conn.cursor()

#create a table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS symbol (
        id     INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL UNIQUE,
        name   TEXT UNIQUE,
        enabled BOOLEAN NOT NULL DEFAULT 'TRUE' CHECK (enabled IN (0, 1)),
        missing_data BOOLEAN NOT NULL DEFAULT 'FALSE' CHECK (missing_data IN (0, 1))
    )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS ohlcv_daily (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        UNIQUE (symbol_id, date)
        FOREIGN KEY (symbol_id) REFERENCES Symbol(id)
    )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS ohlcv_hourly (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        UNIQUE (symbol_id, date)
        FOREIGN KEY (symbol_id) REFERENCES Symbol(id)
    )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS ohlcv_weekly (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        UNIQUE (symbol_id, date)
        FOREIGN KEY (symbol_id) REFERENCES Symbol(id)
    )
''')
cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_hourly_symbol ON ohlcv_hourly(symbol_id)
''')
cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_daily_symbol ON ohlcv_daily(symbol_id)
''')
cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_daily_symbol ON ohlcv_weekly(symbol_id)
''')
cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_Hourly_symbol_date ON ohlcv_hourly(symbol_id, date)
''')
cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_daily_symbol_date ON ohlcv_daily(symbol_id, date)
''')
cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_daily_symbol_date ON ohlcv_weekly(symbol_id, date)
''')
