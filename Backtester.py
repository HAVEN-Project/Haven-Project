import sqlite3
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest
import time
time.sleep(2)
from BackTestingStrategies import *

stock = "NVDA"

# Getting data from Database

conn = sqlite3.connect("StockData.db")

cursor = conn.cursor()

# Get Hourly Data for Chosen stock
sql_query_hourly = "SELECT * FROM DailyData WHERE symbol = ?"

df = pd.read_sql_query(sql=sql_query_hourly, con=conn, params=(stock,))

df = df.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

bt = Backtest(df, Xatrts, cash=10_000, commission=0.002, exclusive_orders=True)

stats = bt.run()

print(stats)

print("\n\n\n")

print(stats['_trades'])

bt.plot()