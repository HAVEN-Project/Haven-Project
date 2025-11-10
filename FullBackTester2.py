import time, os
import yfinance as yf
import pandas as pd
import sqlite3
from types import SimpleNamespace
import re
import numpy as np
import backtrader as bt
import pandas_ta as ta
from datetime import datetime
from backtrader import feeds


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

def GetScoringConfig():
    
    df = pd.read_excel("ScoringConfig.xlsx")

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
    ScoringConfig = SimpleNamespace(**{k: SimpleNamespace(**v) for k, v in nested_params.items()})

    # Access like config.speed.value, config.speed.direction
    print(ScoringConfig)

    return ScoringConfig

#endregion

#region Backtesting logic

class Mark0(bt.Strategy):
    config = GetConfig()
    scoringconfig = GetScoringConfig()
    
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.sma10 = bt.indicators.MovingAverageSimple(self.datas[0], period = 10)
        self.stoch = bt.indicators.Stochastic(self.datas[0])
        self.rsi = bt.indicators.RSI(self.datas[0])
        
        self.order = None

        bt.indicators.ExponentialMovingAverage(self.datas[0], period=25)
        bt.indicators.WeightedMovingAverage(self.datas[0], period=25,
                                            subplot=True)
        bt.indicators.StochasticSlow(self.datas[0])
        bt.indicators.MACDHisto(self.datas[0])
        rsi = bt.indicators.RSI(self.datas[0])
        bt.indicators.SmoothedMovingAverage(rsi, period=10)
        bt.indicators.ATR(self.datas[0], plot=False)


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            self.order = self.buy()
            # Not yet ... we MIGHT BUY if ...
            # if self.data.close[0] > self.sma10[0]:

            #     # BUY, BUY, BUY!!! (with all possible default parameters)
            #     self.log('BUY CREATE, %.2f' % self.dataclose[0])

            #     # Keep track of the created order to avoid a 2nd order
            #     self.order = self.buy()

        else:

            if self.data.close[0] < self.sma10[0]:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()
#endregion

def BackTest(stock, Mark0):
    conn = sqlite3.connect("StockData.db")

    cursor = conn.cursor()

    # Get Daily Data for Chosen stock
    sql_query_daily = "SELECT * FROM DailyData WHERE symbol = ? AND date >= '2023-10-01 00:00:00' ORDER BY date ASC"

    df = pd.read_sql_query(
    sql_query_daily,
    con=conn,
    params=(stock,),
    parse_dates=["date"],
    index_col="date",
)

    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
          
    cerebro = bt.Cerebro()

    cerebro.addstrategy(Mark0)

    df.index = pd.to_datetime(df.index) 
    
    data_feed = feeds.PandasData(
        dataname=df,
    )

    cerebro.adddata(data_feed)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.plot()

ticker = input("Enter Stock to back test: ")

stock = [ticker]

DataDownloader(stock, ticker)

BackTest(ticker, Mark0)