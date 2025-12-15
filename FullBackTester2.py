import time, os
import quantstats as qs
import yfinance as yf
import pandas as pd
import sqlite3
from types import SimpleNamespace
import re
import numpy as np
from backtrader_plotly.plotter import BacktraderPlotly
from backtrader_plotly.scheme import PlotScheme
import backtrader as bt
import pandas_ta as ta
from datetime import datetime
from backtrader import feeds
import plotly.express as px


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

def DataDownloader(tickers):
    
    required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    
    start_date = "2023-10-01"
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
    # cursor.execute("SELECT MAX(date) FROM DailyData WHERE symbol=?", (tickers,))
    # last_date = cursor.fetchone()[0]  # this returns 'YYYY-MM-DD' or None if no data

    # if last_date:
    #     last_date = datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S")
    #     # Download data for all tickers at once
    # else: "max"
        
    attempts = 0
    data = None

    while attempts < 3:
        try:
            # Download with date range
            data = yf.download(tickers , start= start_date, period="max", group_by='ticker', interval=download_interval)
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
    # print(config)

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
    # print(ScoringConfig)

    return ScoringConfig

#endregion

#region Backtester Custom Indicators
class Growth(bt.Indicator):
    '''
    Growth = (data / data[-period]) - 1

    By default uses close price and a 20‑bar lookback.
    '''
    lines = ('growth',)
    params = dict(
        period=20,
        data=None,   # allow custom data line; fallback to self.data
    )

    def __init__(self):
        # choose input line
        data = self.p.data if self.p.data is not None else self.data

        # ensure we have enough bars
        self.addminperiod(self.p.period + 1)

        # vectorized line operation: current / past - 1
        self.l.growth = data / data(-self.p.period) - 1.0

#endregion

#region Backtesting logic

class mark1(bt.Strategy):
    config = GetConfig()
    scoringconfig = GetScoringConfig()
    buyscore = int(scoringconfig.buy.buy)
    sellscore = int(scoringconfig.sell.sell)
    sma10len   = int(config.sma10.length)
    stochKlen  = int(config.stoch.klen)
    stochK     = int(config.stoch.k)
    stochD     = int(config.stoch.d)
    rsilen     = int(config.rsi.length)
    atrtsK     = int(config.atrts.k)
    atrtsLength = int(config.atrts.length)
    atrlength  = int(config.atr.length)

    def __init__(self):
        self.sma10 = []
        for d in self.datas:
            self.sma10.append(bt.indicators.SMA(d.close, period = self.sma10len))
        self.stoch = []
        for d in self.datas:
            self.stoch.append(bt.indicators.Stochastic(d, period = self.stochKlen, period_dfast = self.stochK, period_dslow = self.stochD))
        self.rsi = []
        for d in self.datas:
            self.rsi.append(bt.indicators.RSI(d.close, period = self.rsilen))
        self.atr = []
        for d in self.datas:
            self.atr.append(bt.indicators.ATR(d, period= self.atrlength))
        n = len(self.datas)
        self.insma10dip = [False] * n
        self.dipstart   = [None] * n
        self.diplow     = [None] * n
        print('sma10len', self.sma10len, type(self.sma10len))
        print('stochKlen', self.stochKlen, 'stochK', self.stochK, 'stochD', self.stochD)
        print('rsilen', self.rsilen, 'atrlength', self.atrlength)
        #self.threemonths = Growth(d.data, period = 66)
        #self.oneyear = Growth(d.data, period = 252)
        self.values = []

        self.open_positions = set()          # data indices you own
        self.buy_orders = {}
        self.sell_orders = {}

        self.entry_dates = {d: None for d in self.datas}
        
        self.trades = []   # list of dicts with per-trade info
        self._open_trades = {}
        # thresholds
        self.sell_threshold = -0.3

        self.datetimes = []

        self.asset_value = {d: [] for d in self.datas}
        self.asset_dates = []
        self.portfolio_value = []
        
        self.asset_pnl = {d: [] for d in self.datas}   # running PnL per stock
        self.asset_pnl_dates = []                      # timestamps for PnL updates
        self._pnl_by_data = {d: 0.0 for d in self.datas}  # internal accumulator

        self._last_exec_price = {d: None for d in self.datas}
        self._entry_info = {d: None for d in self.datas} 


    def sma10dips(self, i):
        cur = self.sma10[i][0]
        prev = self.sma10[i][-1]

        # still in / entering dip
        if cur < prev:
            if not self.insma10dip[i]:
                self.insma10dip[i] = True
                self.dipstart[i] = prev
                self.diplow[i] = cur
            else:
                self.diplow[i] = min(self.diplow[i], cur)
            return None  # dip still forming

        # dip ended: compute depth
        if self.insma10dip[i] and cur > prev:
            dip_depth = self.dipstart[i] - self.diplow[i]
            dip_depth_pct = (dip_depth / self.dipstart[i]) * 100

            self.insma10dip[i] = False
            self.diplow[i] = None
            self.dipstart[i] = None
            return dip_depth_pct

        return None

    def compute_score_buy(self, i):
        # Calculates score out of
        score = 0  # This should be a % from 0-100 (ex: 73% match)
        # SMA 10 dip Trend
        dip_pct = self.sma10dips(i)
        if not dip_pct:
            dip_pct = 0
        sma10dip = 0.032  # This is 3.2% by default
        if (sma10dip < dip_pct):
            score += self.scoringconfig.smadip.smadip
    
            # RSI
        if (40 <= self.rsi[i][0] <= 70):
            score += self.scoringconfig.rsi.rsi

        # ATR (Buy signal)
        if (3 < self.atr[i][0] < 4):
            score += self.scoringconfig.atr.atr

        if (self.stoch[i][0] > 20 and self.stoch[i][-1]):
            score += self.scoringconfig.stoch.stoch

        if(self.config.growth.enabled):
            if (self.threemonths[i][0]) > 25:
                score += self.scoringconfig.threemonths.threemonths

            # if self.oneyear[-1] > 2 :
            #     score += self.scoringconfig.oneyear.oneyear
        return score

    def compute_score_sell(self, i):
        # Get Positions
        d = self.datas[i]
        pos = self.getposition(d)
        entry_price = pos.price
        current_price = d.close[0]
        # Calculates score out of
        score = 0  # This should be a % from 0-100 (ex: 73% match)
        #if self.position and self.data.Close[-1] < self.atrts[-1] and self.data.Close[-2] <= self.atrts[-2]:
        #     return 100  # returns 100% match to sell
        if pos.size != 0:
            if entry_price * 1.15 < current_price:
                return 100  # returns 100% match to sell
        # RSI
        if self.rsi[i][0] > 70:
            score += 5

        if self.stoch[i][0] < 70 and self.stoch[i][-1] > 70:
            score += 5  
            
        entry_dt = self.entry_dates.get(d)
        if entry_dt is not None:
            now_dt = self.datas[0].datetime.datetime(0)
            held_days = (now_dt - entry_dt).days
            # if held_days > 14:
            #     score = max(score, 100)  # force strong sell signal

        return score
        
    def next(self):
        scores = []
        for i, d in enumerate(self.datas):
            score = self.compute_score_buy(i)  # per-asset scorer you already made
            scores.append((score, i, d))
        ranked = sorted(scores, reverse=True)

        score_threshold = self.buyscore
        top_n = 5

        filtered = [x for x in ranked if x[0] >= score_threshold]

        top_assets = filtered[:top_n]

        # Keep track of what you already own
        top_is = [x[1] for x in top_assets]

        dt = self.datas[0].datetime.datetime(0)
        self.asset_dates.append(dt)
        self.asset_pnl_dates.append(dt)

        # Per-asset position value (NaN when flat so returns work)
        for d in self.datas:
            self.asset_pnl[d].append(self._pnl_by_data[d])
            pos = self.getposition(d)
            if pos.size != 0:
                value = pos.size * d.close[0]
            else:
                value = float('nan')
            self.asset_value[d].append(value)

        # Portfolio equity
        self.portfolio_value.append(self.broker.getvalue())



        for i, d in enumerate(self.datas):
            pos = self.getposition(d)  # Position object
            # if pos: 
            #     print(pos)
            pos_size = pos.size  # positive -> long, negative -> short, 0 -> flat

            # If this asset is in the top N, ensure we have a long position (20%)
            if i in top_is:
                # Only open a long if we don't already have a long
                if pos_size <= 0:  # covers both flat and existing short (avoid auto-short)
                    # If there's an existing short, explicitly avoid sending a sell to increase short
                    # Instead we set target to +20% which will instruct broker to move to long
                    # (Backtrader will create orders to reach that target; we still are explicit).
                    self.order_target_percent(data=d, target=0.20)
            else:
                # Not in top N: liquidate only if we currently hold a long
                if pos_size > 0:
                    sell_score = self.compute_score_sell(i)

                    # Use order_target_percent to target 0% allocation (liquidate long).
                    # This is safer than self.sell() which could be misused and open a short in some cases.
                    if(sell_score > self.sellscore):
                        self.order_target_percent(data=d, target=0.0)

        # Save portfolio metrics for plotting
        self.values.append(self.broker.getvalue())
        self.datetimes.append(self.datas[0].datetime.datetime(0))
        print("Cash:", self.broker.get_cash())
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        d = trade.data          # the data feed (stock) this trade belongs to

        entry = self._entry_info.get(d) or {}
        entry_size = entry.get("size")
        entry_price = entry.get("price_open")
        price_close = self._last_exec_price.get(d)

        dt_close = self.datas[0].datetime.datetime(0)
        dt_open = self.entry_dates.get(d)

        record = {
            "symbol": d._name,
            "datetime_open": dt_open,
            "datetime_close": dt_close,
            "size": entry_size,
            "total_value": entry_size * entry_price,
            "price_open": entry_price,       # average entry price
            "price_close": price_close,
            "pnl_gross": trade.pnl,
            "pnl_net": trade.pnlcomm,
            "Return %": (((entry_size * price_close) / (entry_size * entry_price)) - 1) * 100,
        }
        self.trades.append(record)

        self._pnl_by_data[d] += trade.pnlcomm  # add net PnL of this trade


    def notify_order(self, order):
        # Use safe checks and avoid referencing non-existent attributes
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            side = 'BUY' if order.isbuy() else 'SELL'
            print(f"Order completed: {order.data._name} Side: {side} Size: {order.size} Price: {order.executed.price}")
            d = order.data
            dt = self.datas[0].datetime.datetime(0)

            self._last_exec_price[d] = order.executed.price

            if order.isbuy():
                print("BUY filled", d._name, "at", dt)
                # store entry info when opening the trade
                self._entry_info[d] = {
                    "size": order.executed.size,
                    "price_open": order.executed.price,
                }
                self.entry_dates[d] = dt
            else:
                # on sell, clear entry date
                self.entry_dates[d] = None

            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"Order FAILED: {getattr(order.data, '_name', 'N/A')} Status: {order.status}")
            
    def stop(self):
        # If you still have an open position, close it
        if self.position.size != 0:
            self.close()        # closes long or short
            print("Closing all open positions at end of backtest.")

#endregion

def BackTest(tickers, Mark0):
    conn = sqlite3.connect("StockData.db")

    cursor = conn.cursor()

    cerebro = bt.Cerebro()

    cerebro.addstrategy(Mark0)

    # Get Daily Data for Chosen stock
    sql_query_daily = "SELECT * FROM DailyData WHERE symbol = ? AND date >= '2023-10-01 00:00:00' ORDER BY date ASC"

    master_data_feed = None

    for ticker in tickers:
        df = pd.read_sql_query(
        sql_query_daily,
        con=conn,
        params=(ticker,),
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
        
            



        df.index = pd.to_datetime(df.index) 
        
        data_feed = feeds.PandasData(
            dataname=df,
            name=ticker
        )
        # print(ticker, df.shape, df.head(), df.tail())

        # Set all other data feeds to plot on the first (master) chart
        # if master_data_feed is None:
        #     master_data_feed = data_feed  # first ticker becomes master
        # else:
        #     data_feed.plotinfo.plotmaster = master_data_feed


        cerebro.adddata(data_feed)


    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)
    cerebro.broker.set_shortcash(False)

    

    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
    # cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    backtest_result = cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Get the strategy instance
    strategy = backtest_result[0]

    df = pd.DataFrame({
    'datetime': strategy.datetimes,
    'portfolio_value': strategy.values
    })

    # asset_equity = {}

    # asset_equity = {
    #     d._name: pd.Series(strategy.asset_value[d], index=pd.to_datetime(strategy.asset_dates))
    #     for d in strategy.datas
    # }
    #     # print(asset_equity)

    # asset_df = pd.DataFrame(asset_equity)
    # # print(asset_df)
    # print(asset_df.columns)

    trades_df = pd.DataFrame(strategy.trades)
    # print(trades_df.head())
    trades_df.to_csv("trades_log.csv", index=False)

    # PnL per Trade
    fig = px.bar(
        trades_df,
        x=trades_df.index,
        y="pnl_net",
        color="symbol",
        title="PnL per Trade"
    )
    fig.show()

    # Portfolio equity curve
    port_series = pd.Series(
        strategy.portfolio_value,
        index=pd.to_datetime(strategy.asset_dates),
        name="portfolio_value"
    )

    # Per-asset position value over time
    asset_equity = {
        d._name: pd.Series(strategy.asset_value[d],
                        index=pd.to_datetime(strategy.asset_dates))
        for d in strategy.datas
    }
    asset_df = pd.DataFrame(asset_equity)

    pnl_equity = {
        d._name: pd.Series(strategy.asset_pnl[d],
                        index=pd.to_datetime(strategy.asset_pnl_dates))
        for d in strategy.datas
    }
    pnl_df = pd.DataFrame(pnl_equity)

    # Portfolio equity
    fig_port = px.line(
        x=port_series.index,
        y=port_series.values,
        title="Portfolio Equity"
    )
    fig_port.show()

    fig_port = px.line(
        pnl_df,
        x=pnl_df.index,
        y=pnl_df.columns,
        title="PnL of each stock"
    )
    fig_port.show()

    # # Per-asset position equity on the same timeline
    # fig_assets = px.line(
    #     asset_df,
    #     x=asset_df.index,
    #     y=asset_df.columns,
    #     title="Per-Asset Position Value"
    # )
    # fig_assets.show()


    # scheme = PlotScheme(decimal_places=2)
    # figs = cerebro.plot(BacktraderPlotly(show=False, scheme=scheme))
    # for run_figs in figs:
    #     for fig in run_figs:
    #         fig.show()

    # fig = px.line(df, x='datetime', y='portfolio_value', title='Portfolio Value Over Time')
    # fig.show()
    
    # Get the returns dictionary and convert to a Pandas DataFrame
    returns_dict = strategy.analyzers.time_return.get_analysis()
    returns_df = pd.DataFrame(list(returns_dict.items()), columns=['date', 'return'])
    returns_df = returns_df.set_index('date')['return']
    returns_df.index = pd.to_datetime(returns_df.index) # Ensure the index is datetime

    qs.reports.html(returns_df, output='backtest_report.html', title='My Strategy Performance')

    # print("asset_df columns:", asset_df.columns)
    # print(asset_df.describe())


    # for col in asset_df.columns:
    #     returns = asset_df[col].pct_change().dropna()
    #     print(f"{col} returns head:\n", returns.head())

    #     # Only skip if truly empty
    #     if returns.empty:
    #         continue

    #     qs.reports.html(
    #         returns,
    #         output=f"{col}_report.html",   # unique file per stock
    #         title=f"{col} Performance"
    #     )


    #   cerebro.plot()

# ticker_num = input("How many stocks to test: ")
# tickers = []
# for i in range(int(ticker_num)):
#     ticker = input("Enter Stock: ")
#     tickers.append(ticker)
# print(tickers)

tickers = ["NVDA","AAPL","GOOG","GOOGL","MSFT","AMZN","AVGO","META","TSLA","WMT","LLY","JPM","V","ORCL","MA","JNJ","XOM","PLTR","NFLX","BAC","ABBV","COST","HD","AMD","PG","GE","UNH","CSCO","KO","CVX","WFC","IBM","MS","CAT","GS","MU","AXP","MRK","CRM","RTX","PM","APP","MCD","TMUS","ABT","TMO","PEP","AMAT","LRCX","C","DIS","LIN","ISRG","QCOM","INTU","BX","INTC","NOW","BLK","UBER","T","TJX","VZ","SCHW","AMGN","BKNG","NEE","ACN","TXN","DHR","APH","ANET","KLAC","BA","SPGI","COF","GILD","ADBE","PFE","UNP","LOW","PGR","BSX","ADI","SYK","PANW","DE","ETN","WELL","MDT","CRWD","PLD","HON","KKR","CB","COP","VRTX","PH","LMT","HCA","CEG","ADP","HOOD","NEM","BMY","CVS","MCK","NKE","CMCSA","CME","MO","DASH","SBUX","ICE","SO","MMC","GD","MMM","DUK","CDNS","WM","MCO","TT","SNPS","DELL","APO","UPS","AMT","USB","PNC","BK","NOC","SHW","MAR","ELV","HWM","ORLY","ABNB","EMR","REGN","RCL","AON","GM","GLW","CTAS","ITW","ECL","WBD","EQIX","CI","WMB","TDG","MNST","COIN","CMI","JCI","MDLZ","CSX","SPG","FCX","TEL","COR","FDX","NSC","RSG","AJG","PWR","HLT","TFC","TRV","CL","ADSK","STX","FTNT","AEP","MSI","WDC","KMI","WDAY","SLB","ROST","EOG","PCAR","SRE","PYPL","AFL","VST","NXPI","BDX","AZO","PSX","ARES","IDXX","MPC","F","ALL","MET","APD","DLR","LHX","NDAQ","O","ZTS","URI","VLO","DDOG","EA","D","GWW","PSA","FAST","EW","ROP","CMG","CAH","CBRE","AME","OKE","BKR","AMP","AIG","DHI","ROK","MPWR","DAL","FANG","CTVA","TTWO","CARR","AXON","XEL","LVS","EXC","TGT","FICO","YUM","ETR","PAYX","MSCI","PRU","TKO","CTSH","OXY","KDP","GRMN","KR","CCI","A","XYZ","PEG","TRGP","VMC","EBAY","GEHC","MLM","IQV","NUE","HIG","EL","CPRT","FISV","HSY","VTR","RMD","WAB","MCHP","CCL","KEYS","STT","SYY","ED","FIS","EQT","UAL","OTIS","KMB","ACGL","WEC","EXPE","XYL","ODFL","PCG","LYV","KVUE","IR","HUM","RJF","FITB","FOXA","HPE","MTB","WTW","NRG","VICI","SYF","TER","VRSK","CHTR","EXR","FOX","LEN","DG","KHC","CSGP","ROL","ADM","IBKR","MTD","HBAN","EME","BRO","TSCO","FSLR","DOV","ATO","EFX","DTE","EXE","BR","ULTA","CBOE","WRB","AEE","NTRS","DXCM","CINF","DLTR","AWK","STZ","FE","ES","BIIB","OMC","TPR","PPL","STLD","CFG","AVB","GIS","STE","CNP","PHM","IRM","VLTO","TDY","LDOS","RF","HAL","LULU","HUBB","EQR","JBL","DVN","PPG","WAT","NTAP","TROW","HPQ","KEY","RL","EIX","VRSN","WSM","ON","CPAY","LH","L","NVR","LUV","CMS","DRI","TSN","PTC","PODD","SBAC","IP","EXPD","CHD","DGX","CNC","CTRA","NI","PFG","TYL","GPN","SMCI","TPL","WST","TRMB","AMCR","JBHT","CDW","INCY","CHRW","PKG","GPC","SNA","ZBH","BG","MKC","LII","TTD","FTV","PNR","ESS","DD","GEN","DOW","APTV","EVRG","GDDY","WY","IT","LNT","HOLX","INVH","IFF","J","COO","MAA","ALB","BBY","TXT","NWS","FFIV","PSKY","ERIE","DECK","NWSA","DPZ","LYB","AVY","UHS","ALLE","EG","KIM","BALL","ZBRA","JKHY","VTRS","IEX","MAS","HRL","NDSN","UDR","HII","HST","WYNN","CLX","BXP","REG","AKAM","CF","BEN","BLDR","ALGN","DOC","SWK","IVZ","EPAM","MRNA","AIZ","HAS","RVTY","CPT","GL","DAY","FDS","SJM","PNW","MGM","SWKS","AES","BAX","AOS","CRL","NCLH","GNRC","TAP","APA","PAYC","TECH","HSIC","POOL","MOH","FRT","CPB","DVA","CAG","MOS","LW","ARE","LKQ","MTCH","MHK"]

# tickers = ["TSLA"]

# DataDownloader(tickers)

BackTest(tickers, mark1)