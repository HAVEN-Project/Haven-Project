import yfinance as yf
import pandas as pd
import sqlite3
import datetime

# Params
today = datetime.date.today()
required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
tickers = ["NVDA", "AAPL", "COIN","SHOP"]
tickers = ["NVDA","AAPL","GOOG","GOOGL","MSFT","AMZN","AVGO","META","TSLA","WMT","LLY","JPM","V","ORCL","MA","JNJ","XOM","PLTR","NFLX","BAC","ABBV","COST","HD","AMD","PG","GE","UNH","CSCO","KO","CVX","WFC","IBM","MS","CAT","GS","MU","AXP","MRK","CRM","RTX","PM","APP","MCD","TMUS","ABT","TMO","PEP","AMAT","LRCX","C","DIS","LIN","ISRG","QCOM","INTU","BX","GEV","INTC","NOW","BLK","UBER","T","TJX","VZ","SCHW","AMGN","BKNG","NEE","ACN","TXN","DHR","APH","ANET","KLAC","BA","SPGI","COF","GILD","ADBE","PFE","UNP","LOW","PGR","BSX","ADI","SYK","PANW","DE","ETN","WELL","MDT","CRWD","PLD","HON","KKR","CB","COP","VRTX","PH","LMT","HCA","CEG","ADP","HOOD","NEM","BMY","CVS","MCK","NKE","CMCSA","CME","MO","DASH","SBUX","ICE","SO","MMC","GD","MMM","DUK","CDNS","WM","MCO","TT","SNPS","DELL","APO","UPS","AMT","USB","PNC","BK","NOC","SHW","MAR","ELV","HWM","ORLY","ABNB","EMR","REGN","RCL","AON","GM","GLW","CTAS","ITW","ECL","WBD","EQIX","CI","WMB","TDG","MNST","COIN","CMI","JCI","MDLZ","CSX","SPG","FCX","TEL","COR","FDX","NSC","RSG","AJG","PWR","HLT","TFC","TRV","CL","ADSK","STX","FTNT","AEP","MSI","WDC","KMI","WDAY","SLB","ROST","EOG","PCAR","SRE","PYPL","AFL","VST","NXPI","BDX","AZO","PSX","ARES","IDXX","MPC","F","ALL","MET","APD","DLR","LHX","NDAQ","O","ZTS","URI","VLO","DDOG","EA","D","GWW","PSA","FAST","EW","ROP","CMG","CAH","CBRE","AME","OKE","BKR","AMP","AIG","DHI","ROK","MPWR","DAL","FANG","CTVA","TTWO","CARR","AXON","XEL","LVS","EXC","TGT","FICO","YUM","ETR","PAYX","MSCI","PRU","TKO","CTSH","OXY","KDP","GRMN","KR","CCI","A","XYZ","PEG","TRGP","VMC","EBAY","GEHC","MLM","IQV","NUE","HIG","EL","CPRT","FISV","HSY","VTR","RMD","WAB","MCHP","CCL","KEYS","STT","SYY","ED","FIS","EQT","UAL","OTIS","KMB","ACGL","WEC","EXPE","XYL","ODFL","PCG","LYV","KVUE","IR","HUM","RJF","FITB","FOXA","HPE","MTB","WTW","NRG","VICI","SYF","TER","VRSK","SNDK","CHTR","EXR","FOX","LEN","DG","KHC","CSGP","ROL","ADM","IBKR","MTD","HBAN","EME","BRO","TSCO","FSLR","DOV","ATO","EFX","DTE","EXE","BR","ULTA","CBOE","WRB","AEE","NTRS","DXCM","CINF","DLTR","AWK","STZ","FE","ES","BIIB","OMC","TPR","PPL","STLD","CFG","AVB","GIS","STE","CNP","PHM","IRM","VLTO","TDY","LDOS","RF","HAL","LULU","HUBB","EQR","JBL","DVN","PPG","WAT","NTAP","TROW","HPQ","KEY","RL","EIX","VRSN","WSM","ON","CPAY","LH","L","NVR","LUV","CMS","DRI","TSN","PTC","PODD","SBAC","IP","EXPD","CHD","DGX","CNC","CTRA","NI","PFG","TYL","GPN","SW","SMCI","TPL","WST","TRMB","AMCR","JBHT","CDW","INCY","CHRW","PKG","GPC","SNA","ZBH","BG","MKC","LII","TTD","FTV","PNR","ESS","DD","GEN","DOW","APTV","EVRG","GDDY","WY","IT","LNT","HOLX","Q","INVH","IFF","J","COO","MAA","ALB","BBY","TXT","NWS","FFIV","PSKY","ERIE","DECK","NWSA","DPZ","SOLV","LYB","AVY","UHS","ALLE","EG","KIM","BALL","ZBRA","JKHY","VTRS","IEX","MAS","HRL","NDSN","UDR","HII","HST","WYNN","CLX","BXP","REG","AKAM","CF","BEN","BLDR","ALGN","DOC","SWK","IVZ","EPAM","MRNA","AIZ","HAS","RVTY","CPT","GL","DAY","FDS","SJM","PNW","MGM","SWKS","AES","BAX","AOS","CRL","NCLH","GNRC","TAP","APA","PAYC","TECH","HSIC","POOL","MOH","FRT","CPB","DVA","CAG","MOS","LW","ARE","SOLS","LKQ","MTCH","MHK"]
start_date = "2023-10-01"
end_date = "2025-10-17"
download_interval = "1d" # hourly (1h) daily is (1d)

# Database configuration
DB_NAME = "StockData.db"
TABLE_NAME = "HourlyData"
if (download_interval == "1d"):
    TABLE_NAME = "DailyData"

data = None 

# data = yf.download(tickers, period = "max", group_by='ticker', interval=download_interval)

# print(data)

conn = sqlite3.connect("StockData.db")
cursor = conn.cursor()

ticker_dfs = []

for ticker in tickers:
    
    cursor.execute("SELECT MAX(date) FROM DailyData WHERE symbol=?", (ticker,))
    last_date = cursor.fetchone()[0]  # this returns 'YYYY-MM-DD' or None if no data
    if(last_date == None):
        last_date = "2020-01-01"
    if(last_date == today):
        print("Data For Stock already Acquired for today")
        continue
    
    print(last_date)



    data = yf.download(ticker,start=last_date, group_by='ticker', interval=download_interval)

    if data:
        #extracts data for the ticker
        ticker_data = data[ticker].copy()

        # Add symbol to column
        ticker_data['symbol'] = ticker
        
        # Reset index to make date a column
        ticker_data = ticker_data.reset_index()

        #rename columns to match required columns format
        ticker_data.columns = ticker_data.columns.str.lower()

        # 1) If any NaN in required columns -> skip whole ticker
        if ticker_data[required_cols].isna().any().any():
            print(f"Skipping {ticker}: missing OHLCV data")
            continue

        # 2) If you also want to enforce *no date gaps* for daily data
        ticker_data = ticker_data.sort_values('date')
        ticker_data['date'] = pd.to_datetime(ticker_data['date'])

        # Build expected calendar from first to last date (business days)
        expected = pd.date_range(
            start=ticker_data['date'].iloc[0],
            end=ticker_data['date'].iloc[-1],
            freq='B'
        )

        if len(expected) != len(ticker_data) or not (ticker_data['date'] == expected).all():
            print(f"Skipping {ticker}: missing trading days")
            continue

        # Used to only inster data newer then this date
        cut_off = last_date

        # If passed all checks, keep only the required columns and write
        available_cols = [col for col in required_cols if col in ticker_data.columns]
        ticker_data = ticker_data[available_cols]
        ticker_data['date'] = ticker_data['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        ticker_data = ticker_data[ticker_data['date']] >= cut_off

        ticker_data.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
        
        
        