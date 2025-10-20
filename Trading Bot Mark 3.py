# Step 0 ~ Libraries
#region import
from sys import maxsize
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
from rich.console import Console
from pyfiglet import figlet_format
from rich.text import Text
from rich.panel import Panel
import time
from datetime import datetime
import sys, select


console = Console(force_terminal=True)
trading_client = TradingClient("PK58DWB3ETACZRPLDKWY","3efM2Riaz99eaM4uqIybfcsoETRa2dorppZbEJkY", paper=True)
#endregion


#region Buying Indicators (INC) means fully written in and operational

#region -------Buying Indicators--------  NOT INTEGRATED YET
bind = {} #Buying Indicators Dictionary

# ---> RSI
bind["useRSI"] = True
bind["minRSI"] = 40
bind["maxRSI"] = 70

# ---> SMA (INC)
bind["useSMA20_50"] = True

# ---> Momentum (INC)
bind["useMomentum"] = False
bind["minMomentum"] = "TBD"

# ---> Month(s) Growth
# (3 Months) (INC)
bind["use3MG"] = True
bind["setMin3Month_Growth"] = 0.18
bind["setMed3Month_Growth"] = 0.23
bind["setHigh3Month_Growth"] = 0.28

# (1 Year)
bind["use1YG"] = True
bind["setMin1yr"] = 0.01
bind["setMed1yr"] = 0.25
bind["setHigh1yr"] = 0.35

# (6 Months)
bind["use6MG"] = False
bind["setMin6MG"] = 0.1
bind["setMed6MG"] = 0.20
bind["setHigh6MG"] = 0.36

# ---> Volatility
bind["useVol"] = True
bind["minVol"] = 0.015
bind["maxVol"] = 0.06

# ---> Test Variables
# ADX (INC)
bind["useADX"] = False
bind["setADX"] = "TBD"

# EMA
bind["useEMA"] = False
bind["minEMA"] = 0.02  # ratio of (EMA_20 / CurrentPrice)

# MACD
bind["useMACD"] = False
# bind["minMACD"] = "TBD"  # not needed currently
# bind["maxMACD"] = "TBD"

# Stochastic Oscillator
bind["useSO"] = False
bind["minSO"] = 20
bind["maxSO"] = 80

# Firm(s) Signals
bind["useFirmSignals"] = False
#endregion #Not integrated ye

#region -----CONFIGURABLE INDICATOR BUYING WINDOWS----- NOT INTEGRATED YET
bwin = {}

bwin["RSI_WINDOW"] = 14
bwin["SMA_SHORT"] = 20
bwin["SMA_LONG"] = 50
bwin["ADX_WINDOW"] = 14
bwin["MOMENTUM_WINDOW"] = 10
bwin["VOLATILITY_WINDOW"] = 20
bwin["EMA_WINDOW"] = 20
bwin["STOCH_WINDOW"] = 14
bwin["STOCH_SMOOTH"] = 3
#endregion

#region-----Configurable Indicator Selling Windows
swin = {}

swin["RSI"] = 14
swin["sma_20"] = 20
swin ["sma_50"] = 50
swin["ADX"] = 14
swin["momentum"] = 10
#endregion

#region--------Selling Indicators-------- NOT INTEGRATED YET
sind = {}

sind["setStopLoss"] = -0.07
sind["setTakeProfit"] = 0.13

# ---> RSI
sind["useRSISell"] = True
sind["maxRSISell"] = 70

# ---> SMA
sind["useSMASell"] = True

# ---> Momentum
sind["useMomentumSell"] = True
sind["minMomentumSell"] = 30

# ---> ADX
sind["useADXSell"] = True
sind["minADXSell"] = 20
run = True

#endregion
#endregion

def score_stock(df, ticker_symbol):


    if df is None or df.empty:
        # no data
        return None

    # Ensure we have enough rows for the biggest lookback used (1 year ~ 240 trading days)
    required_days = 240  # adjust if you change growth windows
    if len(df) < required_days:
        # Data too short for scoring (prevents iloc out-of-bounds)
        return None

    try:


        # Ensure Close column is 1D
        close = df["Close"].squeeze()

        rsi = ta.momentum.RSIIndicator(
            close,
            window=bwin["RSI_WINDOW"]
        ).rsi().iloc[-1]

        # SMA 20 and SMA 50
        sma_20 = close.rolling(window=bwin["SMA_SHORT"]).mean().iloc[-1]
        sma_50 = close.rolling(window=bwin["SMA_LONG"]).mean().iloc[-1]

        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(
            df["High"].squeeze(),
            df["Low"].squeeze(),
            close,
            window=bwin["ADX_WINDOW"]
        ).adx().iloc[-1]

        # Momentum (Rate of Change)
        momentum = ta.momentum.ROCIndicator(
            close,
            window=bwin["MOMENTUM_WINDOW"]
        ).roc().iloc[-1]

        # 3-Month Average Growth (~63 trading days)
        growth_3mo = (close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]

        # 6-Month Average Growth (~126 trading days)
        growth_6mo = (close.iloc[-1] - close.iloc[-126]) / close.iloc[-126]

        # 1-Year Average Growth (~240 trading days)
        growth_1yr = (close.iloc[-1] - close.iloc[-240]) / close.iloc[-240]

        # Volatility (standard deviation of daily returns)
        daily_returns = close.pct_change()
        volatility = daily_returns.rolling(window=bwin["VOLATILITY_WINDOW"]).std().iloc[-1]

        # EMA (Exponential Moving Average)
        ema_20 = ta.trend.EMAIndicator(
            close,
            window=bwin["EMA_WINDOW"]
        ).ema_indicator().iloc[-1]

        # MACD (12 EMA vs 26 EMA)
        macd_line = ta.trend.MACD(close).macd()
        signal_line = ta.trend.MACD(close).macd_signal()

        macd_today = macd_line.iloc[-1]
        signal_today = signal_line.iloc[-1]

        # Stochastic Oscillator (%K)
        stoch = ta.momentum.StochasticOscillator(
            high=df["High"].squeeze(),
            low=df["Low"].squeeze(),
            close=close,
            window=bwin["STOCH_WINDOW"],
            smooth_window=bwin["STOCH_SMOOTH"]
        ).stoch().iloc[-1]
    except (IndexError, KeyError, ValueError) as e:
        # If any indicator failed due to insufficient data / missing columns, skip this ticker
        return None


    # Price filter
    current_price = close.iloc[-1]
    if current_price < 15:
        return None  # skip stock if price < $15

    # Initialize score
    score = 0

    # RSI scoring
    if bind["useRSI"] == True:
        if bind["minRSI"] <= rsi <= bind["maxRSI"]:
            score += 1

    # SMA scoring
    if bind["useSMA20_50"] == True:
        if sma_20 > sma_50:
            score += 1
        else:
            score -= 1

    # ADX scoring (trend strength)
    if bind["useADX"] == True:
        if adx > bind["setADX"]:
            score += 1

    # Momentum scoring
    if bind["useMomentum"] == True:
        if momentum > bind["minMomentum"]:
            score += 1
        else:
            score -= 1

    # 3-month growth scoring
    if bind["use3MG"] == True:
        if growth_3mo > bind["setHigh3Month_Growth"]:
            score += 3
        elif growth_3mo > bind["setMed3Month_Growth"]:
            score += 2
        elif growth_3mo > bind["setMin3Month_Growth"]:
            score += 1
        else:
            score -= 1

    # 6-month growth scoring
    if bind["use6MG"] == True:
        if growth_6mo > bind["setHigh6MG"]:
            score += 3
        elif growth_6mo > bind["setMed6MG"]:
            score += 2
        elif growth_6mo > bind["setMin6MG"]:
            score += 1
        else:
            score -= 1

    # 1-year growth scoring
    if bind["use1YG"] == True:
        if growth_1yr > bind["setHigh1yr"]:
            score += 3
        elif growth_1yr > bind["setMed1yr"]:
            score += 2
        elif growth_1yr > bind["setMin1yr"]:
            score += 1
        else:
            score -= 1

    # Volatility scoring
    if bind["useVol"] == True:
        if bind["minVol"] < volatility < bind["maxVol"]:
            score += 1

    # EMA scoring
    if bind["useEMA"] == True:
        if (ema_20 / current_price) > bind["minEMA"]:
            score += 1

    # MACD scoring
    if bind["useMACD"] == True:
        if macd_today > signal_today:  # MACD above signal → bullish
            score += 1
        elif macd_today < signal_today:  # MACD below signal → bearish
            score -= 1

    # Stochastic oscillator scoring
    if bind["useSO"] == True:
        if stoch < bind["minSO"]:
            score += 1  # oversold → potential bullish
        elif stoch > bind["maxSO"]:
            score -= 1  # overbought → potential bearish


    return score
#This is the start of the exit strategy or the Exit function




def fruit_picker(df, ticker_symbol, buy_price):
    close = df["Close"].squeeze()

    # Indicators
    rsi = ta.momentum.RSIIndicator(close, window=swin["RSI"]).rsi().iloc[-1]
    sma_20 = close.rolling(window=swin["sma_20"]).mean().iloc[-1]
    sma_50 = close.rolling(window=swin ["sma_50"]).mean().iloc[-1]
    adx = ta.trend.ADXIndicator(df["High"].squeeze(), df["Low"].squeeze(), close, window=swin["ADX"]).adx().iloc[-1]
    momentum = ta.momentum.ROCIndicator(close, window=swin["momentum"]).roc().iloc[-1]
    current_price = close.iloc[-1]

    #  Exit thresholds
    stop_loss = sind["setStopLoss"]   # -7% by default
    take_profit = sind["setTakeProfit"]  # +13% by default

    #  Price-based exits
    pct_change = (current_price - buy_price) / buy_price
    if pct_change <= stop_loss:
        #sellPrice = buy_price * 0.93
        return "SELL", f"Stop-loss hit ({pct_change:.2%})", None
    if pct_change >= take_profit:
        sellPrice = buy_price * 1.13
        return "SELL", f"Take-profit hit ({pct_change:.2%})", sellPrice

    # Optional technical exits
    signals = 0
    reasons = []

    if rsi > sind["maxRSISell"]:
        signals += 1
        reasons.append("RSI overbought")

    if sind["useSMASell"]:
        if sma_20 < sma_50:
            signals += 1
            reasons.append("SMA 20 crossed below SMA 50")

    if sind["useMomentumSell"]:
        if momentum < sind["minMomentumSell"]:
            signals += 1
            reasons.append("Momentum turned negative")

    if sind["useADXSell"]:
        if adx < sind["minADXSell"]:
            signals += 1
            reasons.append("Weak trend")

    if signals >= 2:
        return "SELL", ", ".join(reasons), None

    # Default: hold position
    return "HOLD", None, None




#region stocks list
#Stock Lists ----------------Here you enter the stocks you want to analyze (1)
top_10 = []
stock_list_pre = [ #S&P 500 Stocks
    "AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","META","BRK.B","TSLA","UNH",
    "LLY","XOM","JNJ","JPM","V","PG","AVGO","HD","MA","CVX","MRK","ABBV",
    "PEP","COST","KO","BAC","PFE","ADBE"]
#endregion

def buying_and_selling():
    print("\n🔄 Trading algorithm started!\n")
    while True:#--------------> Loops the buy and sell functions


        #Find positions and how many stocks are currently being held
        positions = trading_client.get_all_positions()
        num_stocks_held = len(positions)
        print(f"Currently holding {num_stocks_held} stocks")
        current_stock_list = []
        for pos in positions:
            current_stock_list.append(pos.symbol)

        current_stocks = {
            pos.symbol: {
                "entry_price": float(pos.avg_entry_price),
                "qty": int(pos.qty)
            }
            for pos in positions
        }

        print(current_stocks)





        #Filter for stocks that are already held
        stock_list = list(stock_list_pre)  # make a copy

        for chosenStock in stock_list[:]:  # iterate over a copy to avoid issues
            if chosenStock in current_stock_list:
                stock_list.remove(chosenStock) # check to see if this stock is already owned


        #Declaring dictionary to make each stock in the stocklist hold values -----------------Here the stocks get sorted and ranked (2)
        stock_scores = {}
        if num_stocks_held < 10:
            for stock in stock_list: #For each stock in the list above it will put it through this line of code
                print(f"Processing {stock}...") #Simple message displaying whats being processed in that moment
                try: #It will try to pull this data from every stock (not all data is always available) if it fails it will not crash just move on
                    value = yf.download(stock, period="6mo", interval="1d")
                    score = score_stock(value, stock) #The score variable stores the stock's score after the tests it is put through, it is declared by running the stock through the function above
                    if score is not None:  # skip stocks filtered out
                        stock_scores[stock] = score
                except Exception as e: #In case stocks dont have data, they get skipped
                    print(f"Error with {stock}: {e}")


            sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True) #Grabs the dictionary cretaed earlier and simply sorts it from greatest to least, the lambda part makes sure its grbabing the first value in the dictionary not the actual stock

            print("\nStock Scores:") #After code is finished running, it will print all of the stocks and their scores.
            for stock, score in sorted_stocks:
                print(f"{stock}: {score}")



            #- - - - - - - - - - - > Get infor of how muhc porfolio is worth and how much is left to spend
            account = trading_client.get_account()
            total_value = float(account.equity)
            cash = float(account.cash)
            num_new_stocks = 10 - num_stocks_held
            money_per_stock = cash / num_new_stocks  # allocate cash evenly


            print ( f"\nTotal Porfolio worth: {total_value}"
                    f"\nSpendable Cash: {cash}"
                    f"\nMax cash spend on each new stock purchased: {money_per_stock}"
                    f"\nStocks Held: {num_stocks_held}"
                    f"\nStock Slots Available (out of 10) {num_stocks_held}")



            #Create a list of the top 10 stocks (3)
            top_10 = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)[:10-num_stocks_held] #10 highest scoring stocks
            #current_stocks = dict(current_stock_list)

            if num_stocks_held < 10:
                for stock, score in top_10: #----------------------------->This function grabs the top 10 and places the orders (4)

                    stock_data = yf.Ticker(stock)
                    current_price = stock_data.history(period="1d")["Close"].iloc[-1]
                    qtyi = int(money_per_stock // current_price) #Changed
                    print(f"{stock} current price: {current_price}, qty: {qtyi}")

                    order_request = MarketOrderRequest(

                        symbol = stock,
                        qty = qtyi,  # adjust quantity as desired
                        side = OrderSide.BUY,
                        #time_in_force = TimeInForce.DAY
                        time_in_force = TimeInForce.GTC #(Good till cancelled)
                    )
                    if qtyi == 0:
                        print(f"Not enough cash to buy {stock}, skipping...")
                        stock_list.remove(stock)
                        continue
                    try:
                        order = trading_client.submit_order(order_request)  # <-- this actually submits it
                        print(f"Order submitted for {qtyi} shares of {stock}")

                        #Save a dictionary with stocks and entry prices and quan.
                        #current_stocks[stock] = {
                            #"entry_price": current_price,
                            #"qty": qtyi,


                    except Exception as e:
                        print(f"Failed to submit order for {stock}: {e}")




            else:
                print("  ✅   10 Stocks are currently being held.")

            # Update the stocks being held after purchases so the selling function has correct info
            positions = trading_client.get_all_positions()
            num_stocks_held = len(positions)
            print(f"Currently holding {num_stocks_held} stocks")
            current_stock_list = []
            for pos in positions:
                current_stock_list.append(pos.symbol)

            current_stocks = {
                pos.symbol: {
                    "entry_price": float(pos.avg_entry_price),
                    "qty": int(pos.qty)
                }
                for pos in positions
            }

            print(current_stocks)









        print(f"\n⏰ Checking exits at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if num_stocks_held > 0:
            print (current_stocks)
            for stock in list(current_stocks):

                df = yf.download(stock, period="6mo", interval="1d")
                decision, reason, sellPrice= fruit_picker(df, stock, current_stocks[stock]["entry_price"])

                if decision == "SELL":
                    print(f"Selling {current_stocks[stock]['qty']} shares of {stock} - {reason}")

                    try:
                        # OR if you want to wait until target price:
                        if sellPrice:
                             sell_request = LimitOrderRequest(
                                 symbol = stock,
                                 qty = current_stocks[stock]["qty"],
                                 side = OrderSide.SELL,
                                 limit_price = round(sellPrice, 2),
                                 time_in_force = TimeInForce.GTC
                             )


                        else:
                            sell_request = MarketOrderRequest(
                                symbol = stock,
                                qty = current_stocks[stock]["qty"],
                                side = OrderSide.SELL,
                                time_in_force = TimeInForce.GTC
                            )

                        order = trading_client.submit_order(sell_request)
                        del current_stocks[stock]
                        print(f"✅ Sell order placed for {stock}")




                    except Exception as e:
                        print(f"❌ Failed to sell {stock}: {e}")
        else:
            print("📭 No stocks to check right now.")


        print("Finished Cycle")
        time.sleep(600)
        print("starting cycle")


#region Main Menu
#Main Title
main_title = figlet_format("HAVEN PROJECT", font="slant")
console.print(main_title, style="bright_green")

while True:#MENU
    #region Main Menu
    # print("Hi, Welcome To the TradingBot Version 25.10.19"
    #       "\n\nWhat would you like to do today?\n"
    #       "\n1) Run Trading Bot Algorithm"
    #       "\n2) Current Account Information"
    #       "\n3) Check Current Stocks Held"
    #       "\n4) Sell All Current Stocks (!)"
    #       "\n5) Go Into BackTesting Mode"
    #       "\n6) Run Stock Scoring alone"
    #       "\n7) Adjust Stock Score (Buy)"
    #       "\n8) Adjust Stock Score (Sell)"
    #       "\n9) Run Single Stock Analysis")

    menu_text = Text()
    menu_text.append("1) Run Trading Bot Algorithm\n", style="bright_blue")
    menu_text.append("2) Current Account Information\n", style="bright_yellow")
    menu_text.append("3) Check Current Stocks Held\n", style="bright_blue")
    menu_text.append("4) Sell All Current Stocks (!)\n", style="red")
    menu_text.append("5) Go Into BackTesting Mode\n", style="bright_blue")
    menu_text.append("6) Run Stock Scoring alone\n", style="bright_yellow")
    menu_text.append("7) Adjust Stock Score (Buy)\n", style="bright_blue")
    menu_text.append("8) Adjust Stock Score (Sell)\n", style="bright_yellow")
    menu_text.append("9) Run Single Stock Analysis", style="bright_blue")

    panel = Panel(menu_text, title="[bold yellow]TradingBot Version 25.10.19[/bold yellow]", expand=False)
    console.print(panel)
    #endregion


    userInput = input("\nPlease Enter Your Choice: ")



    if userInput == "1":
        buying_and_selling()
    if userInput == "2":
        account = trading_client.get_account()
        positions = trading_client.get_all_positions()
        num_stocks_held = len(positions)
        #daily_change = float(account.equity) - equity_start_of_day
        #daily_change_percent = daily_change / equity_start_of_day * 100


        print("\n--- Current Account Information ---")
        print(f"Cash/ Buying power: ${account.cash}")
        print(f"Portfolio Value: ${account.portfolio_value}")
        print(f"Day Trade Count: {account.daytrade_count}")
        print(f"Account Status: {account.status}")
        print(f"Number of Stocks Currently Held: {num_stocks_held}")
        print("\nEnter 1 To Return To The Main Menu")
        userInput = input("Selection: ")
        while userInput != "1":
            userInput = input("Please Enter Your Choice: ")
    if userInput == "3":
        print("\nPortfolio Summary:")
        positions = trading_client.get_all_positions()

        if positions:
            total_cost = 0
            total_current = 0

            for p in positions:
                qty = int(p.qty)
                avg = float(p.avg_entry_price)
                current = float(p.current_price)

                cost = avg * qty
                current_value = current * qty

                total_cost += cost
                total_current += current_value

                percent_change = ((current - avg) / avg) * 100
                dollar_change = (current - avg) * qty

                print(f"{p.symbol} | {qty} shares | Avg Buy: {avg:.2f} | Current: {current:.2f} | "
                      f"Percent Up/Down: {percent_change:.2f}% | P/L: ${dollar_change:.2f}")

            if total_cost > 0:
                total_percent_change = ((total_current - total_cost) / total_cost) * 100
                total_dollar_change = total_current - total_cost
                print("\nPortfolio Summary:")
                print(f"Total Cost Basis: ${total_cost:,.2f}")
                print(f"Total Current Value: ${total_current:,.2f}")
                print(f"Total Gain/Loss: ${total_dollar_change:,.2f} ({total_percent_change:.2f}%)")
                print("\n\nEnter 1 To Return To The Main Menu")




            else:
                print("No positions currently held.")



        userInput = input("Selection: ")
        while userInput != "1":
            userInput = input("Please Enter Your Choice: ")
    if userInput == "4":

        print("\n\nARE YOU SURE YOU WOULD LIKE TO EXIT ALL OF YOUR CURRENT POSITIONS?\n")
        print("Enter 563 to exit every position")
        print("Enter 1 To Return To The Main Menu")
        userInput = input("Please Enter Your Choice: ")
        while userInput != "1" and userInput != "563":
            userInput = input("Please Enter Your Choice: ")



        if userInput == "563":
            positions = trading_client.get_all_positions()
            if len(positions) > 0:

                for p in positions:

                    try:

                        sell_request = MarketOrderRequest(
                            symbol=p.symbol,
                            qty=int(float(p.qty)),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC
                        )

                        order = trading_client.submit_order(sell_request)

                        print(f"✅ Sell order placed for {p.symbol}")

                    except Exception as e:
                        print(f"❌ Failed to sell {p.symbol}: {e}")
            else:
                print("📭 No stocks to check right now.")
    #if userInput == "5":
    if userInput == "6":
        selectedStocks = input("Enter the stocks you wish to score (separated by spaces): ")
        stock_list_input = selectedStocks.upper().split()
        stock_scores2 = {}

        for stock in stock_list_input:  #
            print(f"Processing {stock}...")
            try:
                value = yf.download(stock, period="1y", interval="1d")
                score = score_stock(value,stock)
                if score is not None:
                    stock_scores2[stock] = score
            except Exception as e:
                print(f"Error with {stock}: {e}")

        sorted_stocks = sorted(stock_scores2.items(), key=lambda x: x[1], reverse=True)  # Grabs the dictionary cretaed earlier and simply sorts it from greatest to least, the lambda part makes sure its grbabing the first value in the dictionary not the actual stock

        print("\nStock Scores:")  # After code is finished running, it will print all of the stocks and their scores.
        for stock, score in sorted_stocks:
            print(f"{stock}: {score}")
        print ("\nStocks Have Been Sorted!\n")
        print("Enter 1 To Return To The Main Menu")
        userInput = input("Selection: ")
        while userInput != "1":
            userInput = input("Please Enter Your Choice: ")
    if userInput == "7":

        userInput = input("\n1) Adjust Current Stock Indicators In Use"
                          "\n2) Adjust Windows Of Current Indicators"
                          "\n\nPlease Enter Your Choice:")


        if userInput == "1":

            print("\n--- Adjust Scoring Parameters ---\n"
                  "          Press 1 To Exit\n")
            while True:

                if userInput == "1":
                    print("\n--- Adjust Buying Indicator Parameters ---\n"
                          "          Press 1 To Exit\n")



                    bind["useRSI"] = input("Use RSI? (y/n): ").lower()
                    if bind["useRSI"] == "1":
                        bind["useRSI"] = False
                        break
                    bind["useRSI"] = (bind["useRSI"] == "y")
                    if bind["useRSI"]:
                        bind["minRSI"] = float(input(f"Minimum RSI (current {bind['minRSI']}): "))
                        bind["maxRSI"] = float(input(f"Maximum RSI (current {bind['maxRSI']}): "))

                    bind["useSMA20_50"] = input("Use SMA 20/50 crossover? (y/n): ").lower() == "y"

                    bind["useADX"] = input("Use ADX? (y/n): ").lower()
                    if bind["useADX"] == "1":
                        bind["useADX"] = False
                        break
                    bind["useADX"] = (bind["useADX"] == "y")
                    if bind["useADX"]:
                        bind["setADX"] = float(input(f"ADX threshold (current {bind['setADX']}): "))

                    bind["useMomentum"] = input("Use Momentum? (y/n): ").lower()
                    if bind["useMomentum"] == "1":
                        bind["useMomentum"] = False
                        break
                    bind["useMomentum"] = (bind["useMomentum"] == "y")
                    if bind["useMomentum"]:
                        bind["minMomentum"] = float(input(f"Minimum momentum (current {bind['minMomentum']}): "))

                    bind["use3MG"] = input("Use 3-month growth? (y/n): ").lower()
                    if bind["use3MG"] == "1":
                        bind["use3MG"] = False
                        break
                    bind["use3MG"] = (bind["use3MG"] == "y")
                    if bind["use3MG"]:
                        bind["setMin3Month_Growth"] = float(
                            input(f"Minimum 3-month growth (current {bind['setMin3Month_Growth']}): "))
                        bind["setMed3Month_Growth"] = float(
                            input(f"Medium 3-month growth (current {bind['setMed3Month_Growth']}): "))
                        bind["setHigh3Month_Growth"] = float(
                            input(f"High 3-month growth (current {bind['setHigh3Month_Growth']}): "))

                    bind["use6MG"] = input("Use 6-month growth? (y/n): ").lower()
                    if bind["use6MG"] == "1":
                        bind["use6MG"] = False
                        break
                    bind["use6MG"] = (bind["use6MG"] == "y")
                    if bind["use6MG"]:
                        bind["setMin6MG"] = float(input(f"Minimum 6-month growth (current {bind['setMin6MG']}): "))
                        bind["setMed6MG"] = float(input(f"Medium 6-month growth (current {bind['setMed6MG']}): "))
                        bind["setHigh6MG"] = float(input(f"High 6-month growth (current {bind['setHigh6MG']}): "))

                    bind["use1YG"] = input("Use 1-Year growth? (y/n): ").lower()
                    if bind["use1YG"] == "1":
                        bind["use1YG"] = False
                        break
                    bind["use1YG"] = (bind["use1YG"] == "y")
                    if bind["use1YG"]:
                        bind["setMin1yr"] = float(input(f"Minimum 1-Year Growth (current {bind['setMin1yr']}): "))
                        bind["setMed1yr"] = float(input(f"Medium 1-Year Growth (current {bind['setMed1yr']}): "))
                        bind["setHigh1yr"] = float(input(f"High 1-Year Growth (current {bind['setHigh1yr']}): "))

                    bind["useVol"] = input("Factor In Volatility? (y/n): ").lower()
                    if bind["useVol"] == "1":
                        bind["useVol"] = False
                        break
                    bind["useVol"] = (bind["useVol"] == "y")

                    if bind["useVol"]:
                        bind["minVol"] = float(input(f"Minimum Volatility (current {bind['minVol']}): "))
                        bind["maxVol"] = float(input(f"Maximum Volatility (current {bind['maxVol']}): "))

                    bind["useEMA"] = input("Use EMA? (y/n): ").lower()
                    if bind["useEMA"] == "1":
                        bind["useEMA"] = False
                        break
                    bind["useEMA"] = (bind["useEMA"] == "y")
                    if bind["useEMA"]:
                        bind["minEMA"] = float(input(f"Minimum EMA (current {bind['minEMA']}): "))

                    bind["useMACD"] = input("Use MACD? (y/n): ").lower()
                    if bind["useMACD"] == "1":
                        bind["useMACD"] = False
                        break
                    bind["useMACD"] = (bind["useMACD"] == "y")

                    bind["useSO"] = input("Use Stochastic Oscillator (y/n): ").lower()
                    if bind["useSO"] == "1":
                        bind["useSO"] = False
                        break
                    bind["useSO"] = (bind["useSO"] == "y")
                    if bind["useSO"]:
                        bind["minSO"] = float(
                            input(f"Minimum Stochastic Value (0-100) (current {bind['minSO']}): "))
                        bind["maxSO"] = float(
                            input(f"Maximum Stochastic Value (0-100) (current {bind['maxSO']}): "))

                    print("\n✅ Parameters updated!\n")
                    print("Enter 1 To Return To The Main Menu")
                    userInput = input("Selection: ")


                    while userInput != "1":
                        userInput = input("Please Enter Your Choice: ")
                    if userInput == "1":
                        break

        if userInput == "2":
            print("\n-----Adjust Indicator Windows-----"
                  "\n        (Press 1 to exit)")

            while True:
                AdjustRSI = input("Adjust RSI Window (y/n): ").lower()
                if AdjustRSI == "1":
                    AdjustRSI = False
                    break
                AdjustRSI = (AdjustRSI == "y")
                if AdjustRSI:
                    bwin["RSI_WINDOW"] = float(input(f"RSI Window (current {bwin['RSI_WINDOW']}): "))

                # SMA Short Window
                AdjustSMAShort = input("Adjust SMA Short Window (y/n): ").lower()
                if AdjustSMAShort == "1":
                    AdjustSMAShort = False
                    break
                AdjustSMAShort = (AdjustSMAShort == "y")
                if AdjustSMAShort:
                    bwin["SMA_SHORT"] = float(input(f"SMA Short Window (current {bwin['SMA_SHORT']}): "))

                # SMA Long Window
                AdjustSMALong = input("Adjust SMA Long Window (y/n): ").lower()
                if AdjustSMALong == "1":
                    AdjustSMALong = False
                    break
                AdjustSMALong = (AdjustSMALong == "y")
                if AdjustSMALong:
                    bwin["SMA_LONG"] = float(input(f"SMA Long Window (current {bwin['SMA_LONG']}): "))

                # ADX Window
                AdjustADX = input("Adjust ADX Window (y/n): ").lower()
                if AdjustADX == "1":
                    AdjustADX = False
                    break
                AdjustADX = (AdjustADX == "y")
                if AdjustADX:
                    bwin["ADX_WINDOW"] = float(input(f"ADX Window (current {bwin['ADX_WINDOW']}): "))

                # Momentum Window
                AdjustMomentum = input("Adjust Momentum Window (y/n): ").lower()
                if AdjustMomentum == "1":
                    AdjustMomentum = False
                    break
                AdjustMomentum = (AdjustMomentum == "y")
                if AdjustMomentum:
                    bwin["MOMENTUM_WINDOW"] = float(input(f"Momentum Window (current {bwin['MOMENTUM_WINDOW']}): "))

                # Volatility Window
                AdjustVolatility = input("Adjust Volatility Window (y/n): ").lower()
                if AdjustVolatility == "1":
                    AdjustVolatility = False
                    break
                AdjustVolatility = (AdjustVolatility == "y")
                if AdjustVolatility:
                    bwin["VOLATILITY_WINDOW"] = float(
                        input(f"Volatility Window (current {bwin['VOLATILITY_WINDOW']}): "))

                # EMA Window
                AdjustEMA = input("Adjust EMA Window (y/n): ").lower()
                if AdjustEMA == "1":
                    AdjustEMA = False
                    break
                AdjustEMA = (AdjustEMA == "y")
                if AdjustEMA:
                    bwin["EMA_WINDOW"] = float(input(f"EMA Window (current {bwin['EMA_WINDOW']}): "))

                # Stochastic Oscillator Window
                AdjustStoch = input("Adjust Stochastic Oscillator Window (y/n): ").lower()
                if AdjustStoch == "1":
                    AdjustStoch = False
                    break
                AdjustStoch = (AdjustStoch == "y")
                if AdjustStoch:
                    bwin["STOCH_WINDOW"] = float(input(f"Stochastic Window (current {bwin['STOCH_WINDOW']}): "))

                # Stochastic Smooth Window
                AdjustStochSmooth = input("Adjust Stochastic Smooth Window (y/n): ").lower()
                if AdjustStochSmooth == "1":
                    AdjustStochSmooth = False
                    break
                AdjustStochSmooth = (AdjustStochSmooth == "y")
                if AdjustStochSmooth:
                    bwin["STOCH_SMOOTH"] = float(input(f"Stochastic Smooth (current {bwin['STOCH_SMOOTH']}): "))
    if userInput == "8":
        userInput = input("\n1) Adjust Current Stock Indicators In Use"
                          "\n2) Adjust Windows Of Current Indicators"
                          "\n\nPlease Enter Your Choice:")


        if userInput == "1":
            print("\n--- Adjust Scoring Parameters ---\n"
                  "          Press 1 To Exit\n")

            while True:
                defaultStop = sind["setStopLoss"]
                sind["setStopLoss"] = float(input(f"Set Stop Loss (current {sind['setStopLoss']}): "))
                if sind["setStopLoss"] == 1:
                    sind["setStopLoss"] = defaultStop
                    break

                defaultTakeProfit = sind["setTakeProfit"]
                sind["setTakeProfit"] = float(input(f"Set Take Profit (current {sind['setTakeProfit']}): "))
                if sind["setTakeProfit"] == 1:
                    sind["setTakeProfit"] = defaultTakeProfit
                    break

                sind["useRSISell"] = input("Use RSI as a Selling Indicator? (y/n): ").lower()
                if sind["useRSISell"] == "1":
                    sind["useRSISell"] = False
                    break
                sind["useRSISell"] = (sind["useRSISell"] == "y")
                if sind["useRSISell"]:
                    sind["maxRSISell"] = float(input(f"Maximum RSI (current {sind['maxRSISell']}): "))

                sind["useSMASell"] = input("Use SMA as a Selling Indicator? (y/n): ").lower() == "y"

                sind["useMomentumeSell"] = input("Use Momentum as a Selling Indicator? (y/n): ").lower() == "y"
                if sind["useMomentumeSell"]:
                    sind["minMomentumSell"] = float(input(f"Minimum Momentum (current {sind['minMomentumSell']}): "))

                sind["useADXSell"] = input("Use ADX as a Selling Indicator? (y/n): ").lower() == "y"
                if sind["useADXSell"]:
                    sind["minADXSell"] = float(input(f"Minimum ADX (current {sind['minADXSell']}): "))
        if userInput == "2":
            print("\n-----Adjust Indicator Windows-----"
                  "\n        (Press 1 to exit)")

            while True:
                AdjustRSI = input("Adjust RSI Window (y/n): ").lower()
                if AdjustRSI == "1":
                    AdjustRSI = False
                    break
                AdjustRSI = (AdjustRSI == "y")
                if AdjustRSI:
                    swin["RSI"] = float(input(f"RSI Window (current {swin['RSI']}): "))

                # SMA 20
                AdjustSMAShort = input("Adjust SMA 20 Window (y/n): ").lower()
                if AdjustSMAShort == "1":
                    AdjustSMAShort = False
                    break
                AdjustSMAShort = (AdjustSMAShort == "y")
                if AdjustSMAShort:
                    swin["sma_20"] = float(input(f"SMA 20 Window (current {swin['sma_20']}): "))

                # SMA 50
                AdjustSMALong = input("Adjust SMA 50 Window (y/n): ").lower()
                if AdjustSMALong == "1":
                    AdjustSMALong = False
                    break
                AdjustSMALong = (AdjustSMALong == "y")
                if AdjustSMALong:
                    swin["sma_50"] = float(input(f"SMA 50 Window (current {swin['sma_50']}): "))

                # ADX
                AdjustADX = input("Adjust ADX Window (y/n): ").lower()
                if AdjustADX == "1":
                    AdjustADX = False
                    break
                AdjustADX = (AdjustADX == "y")
                if AdjustADX:
                    swin["ADX"] = float(input(f"ADX Window (current {swin['ADX']}): "))

                # Momentum
                AdjustMomentum = input("Adjust Momentum Window (y/n): ").lower()
                if AdjustMomentum == "1":
                    AdjustMomentum = False
                    break
                AdjustMomentum = (AdjustMomentum == "y")
                if AdjustMomentum:
                    swin["momentum"] = float(input(f"Momentum Window (current {swin['momentum']}"))
    if userInput == "9":
        chosenStock = input("What Stock Would You Like To Analyze? ").upper()
        try:
            # Pull historical data
            df = yf.download(chosenStock, period="1y", interval="1d")  # 1 year to cover all growth windows
            close = df["Close"].squeeze()
            high = df["High"].squeeze()
            low = df["Low"].squeeze()

            # === Indicators ===
            rsi = ta.momentum.RSIIndicator(close, window=bwin["RSI_WINDOW"]).rsi().iloc[-1]
            sma_20 = close.rolling(bwin["SMA_SHORT"]).mean().iloc[-1]
            sma_50 = close.rolling(bwin["SMA_LONG"]).mean().iloc[-1]
            ema_20 = ta.trend.EMAIndicator(close, window=bwin["EMA_WINDOW"]).ema_indicator().iloc[-1]

            # MACD is left as is (or you can remove if not used)
            macd_line = ta.trend.MACD(close).macd().iloc[-1]
            signal_line = ta.trend.MACD(close).macd_signal().iloc[-1]

            stoch = ta.momentum.StochasticOscillator(
                high=high,
                low=low,
                close=close,
                window=bwin["STOCH_WINDOW"],
                smooth_window=bwin["STOCH_SMOOTH"]
            ).stoch().iloc[-1]

            momentum = ta.momentum.ROCIndicator(close, window=bwin["MOMENTUM_WINDOW"]).roc().iloc[-1]

            growth_3mo = (close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]
            growth_6mo = (close.iloc[-1] - close.iloc[-126]) / close.iloc[-126]
            growth_1yr = (close.iloc[-1] - close.iloc[-220]) / close.iloc[-220]

            volatility = close.pct_change().rolling(bwin["VOLATILITY_WINDOW"]).std().iloc[-1]
            adx = ta.trend.ADXIndicator(high, low, close, window=bwin["ADX_WINDOW"]).adx().iloc[-1]

            current_price = close.iloc[-1]
        except:
            continue

        print(f"Analysis for {chosenStock}...")
        print(f"Current Price of {chosenStock}: {current_price}")

        if rsi is not None:
            print (f"RSI for {chosenStock}: {rsi}")
        if sma_20 is not None:
            print (f"20 Day SMA for {chosenStock}: {sma_20}")
        if sma_50 is not None:
            print (f"50 Day SMA for {chosenStock}: {sma_50}")
        if ema_20 is not None:
            print (f"20 Day EMA for {chosenStock}: {ema_20}")
        print(f"\nMACD For {chosenStock}------>")

        if macd_line is not None:
            print (f"MACD line for {chosenStock}: {macd_line}")
        if signal_line is not None:
            print (f"Signal line for {chosenStock}: {signal_line}")


        print("\nOther Indicators")
        if stoch is not None:
            print (f"Stochastic Oscillator for {chosenStock}: {stoch}")
        if momentum is not None:
            print (f"Momentum for {chosenStock}: {momentum}")
        if volatility is not None:
            print (f"Volatility for {chosenStock}: {volatility}")
        if adx is not None:
            print (f"ADX for {chosenStock}: {adx}")


        print (f"\nMonthly Growth For {chosenStock}")
        if growth_3mo is not None:
            print (f"\nGrowth 3mo for {chosenStock}: {growth_3mo}")
        if growth_6mo is not None:
            print (f"Growth 6mo for {chosenStock}: {growth_6mo}")
        if growth_1yr is not None:
            print(f"Growth 1 Year for {chosenStock}: {growth_1yr}")

        print("Enter 1 To Return To The Main Menu")
        userInput = input("Selection: ")
        while userInput != "1":
            userInput = input("Please Enter Your Choice: ")
#endregion



















