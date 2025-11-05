import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime
from types import SimpleNamespace
import re
from backtesting import Strategy
import numpy as np
import pandas_ta as ta
from backtesting import Backtest
import time

#region DataBase Creation

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

conn.close()
#endregion

#region DataDownload

def DataDownloader(tickers, ticker):
    
    required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    
    start_date = "2023-10-18"
    end_date = "2025-10-17"
    download_interval = "1d" # hourly (1h) daily is (1d)

    # Database configuration
    DB_NAME = "StockData.db"
    TABLE_NAME = "HourlyData"
    if (download_interval == "1d"):
        TABLE_NAME = "DailyData"

    conn = sqlite3.connect("StockData.db")
    cursor = conn.cursor()

    # Example: get the last date we have for a stock
    cursor.execute("SELECT MAX(date) FROM DailyData WHERE symbol=?", (ticker,))
    last_date = cursor.fetchone()[0]  # this returns 'YYYY-MM-DD' or None if no data

    if last_date:
        last_date = datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S")
        # Download data for all tickers at once
    else: "max"
        
    attempts = 0
    data = None

    while attempts < 3:
        try:
            # Download with date range
            data = yf.download(ticker ,start=last_date, period="max", group_by='ticker', interval=download_interval)
            print("Download successful!")
            break
        except Exception as e:
            attempts += 1
            print(f"Attempt {attempts} failed: {e}")
            if attempts >= 3:
                print("Failed to download data after 3 attempts")

    # If download was successful, process and insert the data
    if data is not None and not data.empty:
        # Create a list to store individual ticker DataFrames
        ticker_dfs = []
        
        for ticker in tickers:
            # Extract data for this ticker
            ticker_data = data[ticker].copy()
            
            # Add symbol column
            ticker_data['symbol'] = ticker
            
            # Reset index to make date a column
            ticker_data = ticker_data.reset_index()
            
            # DEBUG: Show original columns before processing
            print(f"\n{ticker} original columns after reset_index:", ticker_data.columns.tolist())
            
            # Rename columns to match required_cols format
            ticker_data.columns = ticker_data.columns.str.lower()
            
            # DEBUG: Show columns after lowercasing
            print(f"{ticker} columns after lowercase:", ticker_data.columns.tolist())
            
            # Handle different possible date column names
            if 'date' not in ticker_data.columns:
                if 'datetime' in ticker_data.columns:
                    ticker_data = ticker_data.rename(columns={'datetime': 'date'})
                    print(f"{ticker}: Renamed 'datetime' to 'date'")
                elif 'index' in ticker_data.columns:
                    ticker_data = ticker_data.rename(columns={'index': 'date'})
                    print(f"{ticker}: Renamed 'index' to 'date'")
            
                    # Shift timestamp back 4 hours
            if 'date' in ticker_data.columns:
                ticker_data['date'] = pd.to_datetime(ticker_data['date']) - pd.Timedelta(hours=4)
                print(f"{ticker}: Shifted timestamp back 4 hours")
            
            # Rename adj close if it exists
            if 'adj close' in ticker_data.columns:
                ticker_data = ticker_data.rename(columns={'adj close': 'adjclose'})
            
            # DEBUG: Show final columns
            print(f"{ticker} final columns:", ticker_data.columns.tolist())
            
            ticker_dfs.append(ticker_data)
        
        # Combine all ticker DataFrames
        final_df = pd.concat(ticker_dfs, ignore_index=True)
        
        # DEBUG: Show columns before filtering
        print("\n" + "="*50)
        print("Combined DataFrame columns:", final_df.columns.tolist())
        print("="*50)
        
        # Remove rows with missing data
        rows_before = len(final_df)
        
        # Option 2: Remove rows where specific important columns have missing data (uncomment if preferred)
        final_df = final_df.dropna(subset=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        
        rows_after = len(final_df)
        rows_removed = rows_before - rows_after
        
        if rows_removed > 0:
            print(f"\n✓ Removed {rows_removed} rows with missing data")
            print(f"  Rows before: {rows_before}")
            print(f"  Rows after: {rows_after}")
        else:
            print("\n✓ No rows with missing data found")
        
        # Reorder columns to match required_cols
        available_cols = [col for col in required_cols if col in final_df.columns]
        
        # DEBUG: Show which columns are available
        print("\nRequired columns:", required_cols)
        print("Available columns:", available_cols)
        print("Missing columns:", [col for col in required_cols if col not in final_df.columns])
        
        final_df = final_df[available_cols]
        
        # DEBUG: Show final DataFrame columns
        print("\nFinal DataFrame columns:", final_df.columns.tolist())
        
        # Convert date to string format for database (only if date column exists)
        if 'date' in final_df.columns:
            final_df['date'] = pd.to_datetime(final_df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
            print("✓ Date column converted successfully")
        else:
            print("✗ ERROR: 'date' column not found in final DataFrame!")
            print("Cannot proceed with database insertion without date column")
        
        print("\nFinal DataFrame shape:", final_df.shape)
        print("\nFirst few rows:")
        print(final_df.head())
        
        # Only insert if we have a date column
        if 'date' in final_df.columns:
            # Insert into database
            try:
                # Connect to SQLite database (creates it if it doesn't exist)
                conn = sqlite3.connect(DB_NAME)
                
                # Insert data into database
                # if_exists options: 'fail', 'replace', 'append'
                final_df.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
                
                print(f"\n✓ Successfully inserted {len(final_df)} rows into {TABLE_NAME} table")
                
                # Verify insertion
                cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
                total_rows = cursor.fetchone()[0]
                print(f"✓ Total rows in database: {total_rows}")
                
                # Show sample of inserted data
                cursor.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 5")
                print("\nSample from database:")
                for row in cursor.fetchall():
                    print(row)
                
                conn.close()
                
            except Exception as e:
                print(f"\n✗ Database insertion failed: {e}")
                if 'conn' in locals():
                    conn.close()
        else:
            print("\nSkipping database insertion due to missing date column")
    else:
        print("No data downloaded")

#endregion

#region Get Back Tester Config

def GetConfig():
    
    df = pd.read_excel("BackTesterConfig.xlsx")

    # Normalize Enabled values to boolean
    df["Enabled"] = df["Enabled"].astype(str).str.lower().isin(["true", "1.0", "yes"])

    nested_params = {}

    for _, row in df.iterrows():
        param = row["Parameter"]
        param_clean = re.sub(r'\W|^(?=\d)', '_', str(param).strip().lower())

        subparam = row["Subparameter"]
        value = row["Value"] if pd.notnull(row["Value"]) else row["Default"]
        enabled = row["Enabled"]

        # --- Auto-convert TRUE/FALSE to booleans ---
        if isinstance(value, str) and value.strip().lower() in ["true", "false", "yes", "no", "1", "0"]:
            value = value.strip().lower() in ["true", "yes", "1"]

        # --- Build nested dict ---
        if param_clean not in nested_params:
            nested_params[param_clean] = {"enabled": enabled}
        # Add Params under that param
        nested_params[param_clean][subparam] = value
    # Convert to nested objects
    config = SimpleNamespace(**{k: SimpleNamespace(**v) for k, v in nested_params.items()})

    # Access like config.speed.value, config.speed.direction
    print(config)

    return config

#endregion

#region Backtesting Logic

class Mark0(Strategy):
    config = GetConfig()
    smaShort = config.sma.short
    smaLong = config.sma.long
    emaShort = config.ema.short
    emaLong = config.ema.long
    rsi = config.rsi.length
    adx = config.adx.length
    
    def init(self):
        # getting data
        open = pd.Series(self.data.Open)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        volume = pd.Series(self.data.Volume)

        # Setting Up config
        self.smaShort = Mark0.smaShort
        #indicators
        if(self.config.sma.enabled):
            self.sma20 = self.I(lambda x: ta.sma(x, length=int(self.smaShort)), close)
            self.sma50 = self.I(lambda x: ta.sma(x, length=int(Mark0.smaLong)), close)

        if(self.config.ema.enabled):
            self.ema12 = self.I(lambda x: ta.ema(x, length=int(Mark0.emaShort)), close)
            self.ema26 = self.I(lambda x: ta.ema(x, length=int(Mark0.emaLong)), close)
        
        if(self.config.rsi.enabled):
            self.rsi = self.I(lambda x: ta.rsi(x, length=int(Mark0.rsi)), close)

        if(self.config.adx.enabled):
            self.adx = self.I(lambda x, y, z: ta.adx(x, y, z, length=int(Mark0.adx)), high, low, close)

        #Score
        self.score_indicator = self.I(lambda: np.full_like(self.data.Close, np.nan))

    def compute_score (self):
        #Calculates score out of 
        score = 0
        #SMA Trend
        if(self.config.sma.enabled):
            if(self.data.Close[-1] > self.sma20[-1]): score +=10
            if(self.data.Close[-1] > self.sma50[-1]): score +=10

        #SMA Cross
            if (self.sma20[-1]> self.sma50[-1]): score +=10

        #EMA Trend
        if(self.config.ema.enabled):
            if (self.ema12[-1] > self.ema26[-1]): score +=10

        #RSI
        if(self.config.rsi.enabled):
            if ((self.rsi[-1] >= 40) & (self.rsi[-1] <= 70)): score += 10
            if ((self.rsi[-1] >= 50) & (self.rsi[-1] <= 60)): score += 10
            if (self.rsi[-1] > 80): score -5

        #ADX
        if(self.config.adx.enabled):
            if (self.adx[-1] > 25): score += 10
            if (self.adx[-1] > 40): score += 5

        return score

    def next(self):
        current_score = 0
        current_score = self.compute_score()
        self.score_indicator[-1] = current_score
        if (current_score > 49): 
            self.position.close()
            self.buy()
        elif (current_score < 21): 
            self.position.close()
            self.sell()

class Mark1: 
    config = GetConfig()
    sma10len = config.sma10.length
    stochKlen = config.stoch.klen
    stochK = config.stoch.k
    stochD = config.stoch.d
    rsilen = config.rsi.length


    def init(self):
        
        # getting data
        open = pd.Series(self.data.Open)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        volume = pd.Series(self.data.Volume)

        #indicators

        #sma
        if(self.config.sma.enabled):
            self.sma10 = self.I(lambda x: ta.sma(x, length=int(self.sma10len)), close)
        #stoch
        if(self.config.stoch.enabled):
            self.stoch = self.I(lambda x, y, z: ta.stoch(x, y, z, k=int(self.stochK), d=int(self.stochD), smooth_k=int(self.stochKlen)), high, low, close)
        #rsi
        if(self.config.rsi.enabled):
            self.rsi = self.I(lambda x: ta.rsi(x, length=int(self.rsilen)), close)
        #atrts
        if(self.config.atrts.enabled):
            self.atrts = self.I(lambda h, l, c: ta.atrts(h, l, c, length=14, k=3.5), high, low, close)

    def compute_score_buy(self):
        # Calculates score out of
        score = 0  # This should be a % from 0-100 (ex: 73% match)
        # SMA 10 dip Trend
        sma10dip = 0.032  # This is 3.2% by default
        if "{sma10 dips {sma10dip}":
            score += {"excel grabs sma10 weight out of 100"}

            # RSI
        if ((self.rsi[-1] >= 40) & (self.rsi[-1] <= 70)):
            score += {"excel grabs RSI weight out of 100"}

        # ATR (Buy signal)
        if 3 < {atr} < 4
            score += {"excel grabs ATR weight out of 100"}

        if stochastic crosses 20 from the bottom:
            score += {"excel grabs stoch weight out of 100"}

        if 3mogrowth is greater than 25 %:
            score += {"excel grabs 3mogrowth weight out of 100"}

        if 1 yrgrwth is greater than 2 %:
            score += {"excel grabs 1yrgrwth weight out of 100"}
        return score

    def compute_score_sell(self):
        # Calculates score out of
        score = 0  # This should be a % from 0-100 (ex: 73% match)
        if self.position and price < self.atrts[-1]:
            return 100  # returns 100% match to sell
        if current profit > 15 %:
            return 100  # returns 100% match to sell
        # RSI
        if RSI > 70:
            score += {"excel grabs RSI weight out of 100"}

        if stochastic crosses 80 from the top:
            score += {"excel grabs stoch weight out of 100"}

    def next(self):
        apple = apple

class Mark1(Strategy):


#endregion

#region Runs Backtest

def BackTest(stock, MyStrategy):

    # Getting data from Database

    conn = sqlite3.connect("StockData.db")

    cursor = conn.cursor()

    # Get Daily Data for Chosen stock
    sql_query_daily = "SELECT * FROM DailyData WHERE symbol = ? AND date >= '2023-10-01 00:00:00'"

    df = pd.read_sql_query(sql=sql_query_daily, con=conn, params=(stock,))

    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    bt = Backtest(df, MyStrategy, cash=10_000, commission=0.002, exclusive_orders=True, finalize_trades=True)

    stats = bt.run()

    print(stats)

    print("\n\n\n")

    print(stats['_trades'])

    stats.to_csv('output_pandas.csv', index=True)

    stats['_trades'].to_csv('output_pandas_trades.csv', index=True)

    bt.plot()

    conn.close()

#endregion
ticker = input("Enter Stock to back test: ")

stock = [ticker]

DataDownloader(stock, ticker)

BackTest(ticker, Mark1)
