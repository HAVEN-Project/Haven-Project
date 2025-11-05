import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime

required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
tickers = ["TSLA", "NVDA", "GOOGL", "SPY", "AAPL"]
start_date = "2023-10-18"
end_date = "2025-10-17"
download_interval = "1d" # hourly (1h) daily is (1d)

# Database configuration
DB_NAME = "StockData.db"
TABLE_NAME = "HourlyData"
if (download_interval == "1d"):
    TABLE_NAME = "DailyData"

# Download data for all tickers at once
attempts = 0
data = None

while attempts < 3:
    try:
        # Download with date range
        data = yf.download(tickers, period="max", group_by='ticker', interval=download_interval)
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
            cursor = conn.cursor()
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