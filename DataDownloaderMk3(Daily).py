import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, date, timedelta, time
import logging 
from logging.handlers import RotatingFileHandler


#Logging
logger = logging.getLogger("Daily_Downloader")
logger.setLevel(logging.INFO)
logger.propagate = False

#Formats the logger
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

file_handler = RotatingFileHandler(
    "download.log",
    maxBytes=5*1024*1024,
    backupCount=100
)

#Adds handlers
console_handler = logging.StreamHandler() # Makes output to console
console_handler.setFormatter(formatter) # Formats the logs
file_handler.setFormatter(formatter) #formats logs
logger.addHandler(console_handler) #adds the handler to the logging system (Activate for console logging)
logger.addHandler(file_handler) #Adds the handler that outputs to logs



logger.info("Daily download cycle started")


# Params
required_cols = ['symbol_id', 'date', 'open', 'high', 'low', 'close', 'volume']
error_count = 0

try:
    # Database configuration
    conn = sqlite3.connect("TestStockData.db")
    cursor = conn.cursor()
    SYMBOL_TABLE = "symbol"
    DAILY_TABLE = "ohlcv_daily"
    interval = "1d"

except Exception:
    logger.exception("Failed Database Connection")


#Time Functions
try:
    today = date.today()
    date_format = "%Y-%m-%d"
    current_time = datetime.now().time()
    target_time = time(hour=17, minute=30)
    # today = datetime.strptime(today, "%Y-%m-%d %H-:M:%S")
except Exception:
    logger.exception("Failed During Time Functions")

# Get Ticker List Sorts through for only Enabled Stocks
try:
    tickers = []
    cursor.execute("SELECT ticker FROM symbol WHERE enabled=1 AND missing_data=0" )
    tickers = cursor.fetchall()
    total_tickers = len(tickers)
    if tickers == None:
        raise Exception
except Exception:
    logger.exception("Failed to acquire enabled stocks from Database")

# Gets Row Id
def get_symbol_id(conn, ticker):
    cur = conn.cursor()
    cur.execute("SELECT id FROM symbol WHERE ticker = ?", (ticker))
    row = cur.fetchone()
    if row:
        return row[0]
    # Insert if not exists
    cur.execute("INSERT INTO symbol (ticker) VALUES (?)", (ticker,))
    conn.commit()
    return cur.lastrowid

for ticker in tickers:
    logger.debug(f"Ticker: {ticker}")
    try:
        id = get_symbol_id(conn, ticker)
        logger.debug(f"Stock ID is: {id}")
        if id == None:
            raise Exception
    except Exception:
        logger.exception("Failed to acquire Ticker ID")
        error_count += 1
        continue

    try:
        cursor.execute(f"SELECT MAX(date) FROM {DAILY_TABLE} WHERE symbol_id=?", (id,))
        last_date = cursor.fetchone()[0]  # this returns 'YYYY-MM-DD' or None if no data
    except Exception:
        logger.exception(f"Failed to acquire last date of {ticker}")
        error_count += 1
        continue
    # lastdate = datetime.date(last_da)
    # today = datetime.strptime(today, date_format).date()

    try:
        if(last_date == None):
            last_date = "2020-01-01"
        last_date = datetime.strptime(last_date, date_format).date()
        if isinstance(today, str):
            today = datetime.strptime(today, date_format).date()
        logger.debug(f"today is {today} |  last date is {last_date}  |   ")
        if(last_date == today or (current_time < target_time and (today - timedelta(days=1) == last_date) and today.weekday() != 0)):
            # print(f"Data For {ticker} already Acquired for today. ID: {id}")
            logger.error(f"Data For {ticker} already Acquired for today. ID: {id}")
            error_count += 1
            continue
        next_date = last_date + timedelta(days=1)
    except Exception:
        logger.exception("Failed Time Comparision block")
        error_count += 1
        continue

    try:
        data = yf.download(ticker,start=next_date, group_by='ticker', interval="1d")
        if data is  None or data.empty:
            raise Exception
            
    except Exception:
        logger.exception(f"Failed to download data through Yfinance for {ticker}")
        error_count += 1
        continue

    if data is not None and not data.empty:
        #extracts data for the ticker
        ticker_data = data[ticker].copy()

        # Add symbol to column
        ticker_data['symbol_id'] = id
        
        # Reset index to make date a column
        ticker_data = ticker_data.reset_index()

        #rename columns to match required columns format
        ticker_data.columns = ticker_data.columns.str.lower()

        # 1) If any NaN in required columns -> skip whole ticker
        if ticker_data[required_cols].isna().any().any():
            logger.error(f"Skipping {ticker}: missing OHLCV data")
            error_count += 1
            continue

        # 2) If you also want to enforce *no date gaps* for daily data
        ticker_data = ticker_data.sort_values('date')
        ticker_data['date'] = pd.to_datetime(ticker_data['date'])

        # Used to only insert data newer then this date
        cut_off = pd.to_datetime(last_date)

        # If passed all checks, keep only the required columns and write
        available_cols = [col for col in required_cols if col in ticker_data.columns]
        ticker_data = ticker_data[available_cols]
        ticker_data['date'] = pd.to_datetime(ticker_data['date'])
        ticker_data = ticker_data[ticker_data['date'] >= cut_off]
        ticker_data['date'] = ticker_data['date'].dt.strftime('%Y-%m-%d')

        try:
            ticker_data.to_sql(DAILY_TABLE, conn, if_exists='append', index=False)
        except Exception:
            logger.exception(f"Failed to insert {ticker} to {DAILY_TABLE} | ID: {id}")
            error_count += 1
            continue
        logger.info(f"Finished acquiring data for {ticker} with ID: {id}")

logger.info(f"Finished Daily download cycle for {total_tickers} stocks. With {error_count} errors.")