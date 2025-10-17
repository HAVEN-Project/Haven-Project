# Step 0 ~ Libraries
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


import time
from datetime import datetime
import sys, select


trading_client = TradingClient("PK58DWB3ETACZRPLDKWY","3efM2Riaz99eaM4uqIybfcsoETRa2dorppZbEJkY", paper=True)



#Buying Indicators (INC) means fully written in and operational
#---> RSI (INC)
useRSI = True
minRSI = 40
maxRSI = 70
#--->SMA (INC)
useSMA20_50 = True
#--->Momentum (INC)
useMomentum = False
minMomentum = "TBD"
#--->Month(s) Growth
    #(3 Months) (INC)
use3MG = True
setMin3Month_Growth = 0.18
setMed3Month_Growth = 0.23
setHigh3Month_Growth = 0.28
    #(1 year)
use1YG = True
setMin1yr = 0.01
setMed1yr = 0.25
setHigh1yr = 0.35
    #(6 Months)
use6MG = False
setMin6MG = 0.1
setMed6MG = 0.20
setHigh6MG = 0.36
#hello
#--->Volatility
useVol = True
minVol = 0.015
maxVol = 0.06

#-------Test Variables--------------->
#--->ADX (INC)
useADX = False
setADX = "TBD"
#--->EMA
useEMA = False
minEMA = 0.02 # This is the ratio of (EMA_20/CurrentPrice) Shows if there is a bullish price

#--->MAC-D
useMACD = False
#minMACD = "TBD" (For now we dont need min or max settings for macD)
#maxMACD = "TBD"
#--->Stochastic Oscillator
useSO = False
minSO = 20
maxSO = 80

#Firm(s) Signals
useFirmSignals = False

#------------------------------------------->
#Selling Indicators
setStopLoss = -0.07
setTakeProfit = 0.13
useRSISell = True
maxRSISell = 70
useSMASell = True
useMomentumeSell = True
minMomentumSell = 30
useADXSell = True
minADXSell = 20

run = True

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

        # RSI (Relative strength index)
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

        # SMA 20 and SMA 50
        sma_20 = close.rolling(window=20).mean().iloc[-1]
        sma_50 = close.rolling(window=50).mean().iloc[-1]

        # ADX (Average directional Index
        adx = ta.trend.ADXIndicator(df["High"].squeeze(), df["Low"].squeeze(), close, window=14).adx().iloc[-1]

        # Momentum (using Rate of Change as a substitute for MomentumOscillator) Uses derivative of current movement compared to derivative 10 days ago
        momentum = ta.momentum.ROCIndicator(close, window=10).roc().iloc[-1]

        # 3-month average growth
        growth_3mo = (close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]  # ~63 trading days = 3 months

        # 3-month average growth
        growth_6mo = (close.iloc[-1] - close.iloc[-126]) / close.iloc[-126]  # ~126 trading days = 6 months

        #1- Year average growth
        growth_1yr = (close.iloc[-1] - close.iloc[-240]) / close.iloc[-240]  # ~240 trading days = 12 months


        #VOLATILITY (For now this is based on the standard deviation of returns over last 20 days)
        daily_returns = close.pct_change()
        volatility = daily_returns.rolling(window=20).std().iloc[-1]

        #EMA (Exponential moving average for 20 days)
        ema_20 = ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1]

        #MACD (difference between 12 EMA and 26 EMA) (What to look for: 🔼 MACD crosses above the signal line → bullish momentum (stock gaining strength). 🔽 MACD crosses below the signal line → bearish momentum (stock losing steam). The distance between the lines indicates the strength of the move.)
        macd_line = ta.trend.MACD(close).macd()
        signal_line = ta.trend.MACD(close).macd_signal()

        macd_today = macd_line.iloc[-1]
        signal_today = signal_line.iloc[-1]

        #Stochastic Oscillator (%K)
        stoch = ta.momentum.StochasticOscillator(
            high=df["High"].squeeze(),
            low=df["Low"].squeeze(),
            close=close,
            window=14,
            smooth_window=3
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
    if useRSI == True:
        if minRSI <= rsi <= maxRSI:
            score += 1

    # SMA scoring
    if useSMA20_50 == True:
        if sma_20 > sma_50:
            score += 1
        else:
            score -= 1

    # ADX scoring (trend strength)
    if useADX == True:
        if adx > setADX:
            score += 1

    # Momentum scoring
    if useMomentum == True:
        if momentum > minMomentum:
            score += 1
        else:
            score -= 1

    # 3-month growth scoring
    if use3MG == True:
        if growth_3mo > setHigh3Month_Growth:
            score += 3
        elif growth_3mo > setMed3Month_Growth:
            score += 2
        elif growth_3mo > setMin3Month_Growth:
            score += 1
        else:
            score -= 1

    # 6-month growth scoring
    if use6MG == True:
        if growth_6mo > setHigh6MG:
            score += 3
        elif growth_6mo > setMed6MG:
            score += 2
        elif growth_6mo > setMin6MG:
            score += 1
        else:
            score -= 1


    # 1-year growth scoring
    if use1YG == True:
        if growth_1yr > setHigh1yr:
            score += 3
        elif growth_1yr > setMed1yr:
            score += 2
        elif growth_1yr > setMin1yr:
            score += 1
        else:
            score -= 1
    # Volatility Scoring
    if useVol == True:
        if minVol< volatility < maxVol:
            score += 1
    #EMA Scoring
    if useEMA == True:
        if (ema_20/current_price) > minEMA:
            score += 1
    #MacD Scoring
    if useMACD == True:
        if macd_today > signal_today:  # MACD above signal → bullish
            score += 1
        elif macd_today < signal_today:  # MACD below signal → bearish
            score -= 1

    #Stochastic oscillator scoring
    if useSO == True:
        if stoch < minSO:
            score += 1  # oversold → potential bullish
        elif stoch > maxSO:
            score -= 1  # overbought → potential bearish


    return score












#This is the start of the exit strategy





#Exit function

def fruit_picker(df, ticker_symbol, buy_price):
    close = df["Close"].squeeze()

    # Indicators
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    sma_20 = close.rolling(window=20).mean().iloc[-1]
    sma_50 = close.rolling(window=50).mean().iloc[-1]
    adx = ta.trend.ADXIndicator(df["High"].squeeze(), df["Low"].squeeze(), close, window=14).adx().iloc[-1]
    momentum = ta.momentum.ROCIndicator(close, window=10).roc().iloc[-1]
    current_price = close.iloc[-1]

    #  Exit thresholds
    stop_loss = setStopLoss   # -7% by default
    take_profit = setTakeProfit  # +13% by default

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

    if rsi > maxRSISell:
        signals += 1
        reasons.append("RSI overbought")
    if useSMASell:
        if sma_20 < sma_50:
            signals += 1
            reasons.append("SMA 20 crossed below SMA 50")
    if useMomentumeSell:
        if momentum < 0:
            signals += 1
            reasons.append("Momentum turned negative")
    if useADXSell:
        if adx < 20:
            signals += 1
            reasons.append("Weak trend")

    if signals >= 2:
        return "SELL", ", ".join(reasons), None

    # Default: hold position
    return "HOLD", None, None





















#Stock Lists ----------------Here you enter the stocks you want to analyze (1)
top_10 = []

stock_list_pre = [ #S&P 500 Stocks
    "AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","META","BRK.B","TSLA","UNH",
    "LLY","XOM","JNJ","JPM","V","PG","AVGO","HD","MA","CVX","MRK","ABBV",
    "PEP","COST","KO","BAC","PFE","ADBE"]

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






#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#Note: This is where the actual program code starts, nothing to do with the trading algorithm, that is all above.


while True:

    print("\nHi, Welcome To the TradingBot Version 25.10.2"
          "\n\nWhat would you like to do today?\n"
          "\n1) Run Trading Bot Algorithm"
          "\n2) Current Account Information"
          "\n3) Check Current Stocks Held"
          "\n4) Sell All Current Stocks (!)"
          "\n5) Go Into BackTesting Mode"
          "\n6) Run Stock Scoring alone"
          "\n7) Adjust Stock Score (Buy)"
          "\n8) Adjust Stock Score (Sell)"
          "\n9) Run Single Stock Analysis")


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

        print("\n--- Adjust Scoring Parameters ---\n")

        useRSI = input("Use RSI? (y/n): ").lower() == "y"
        if useRSI:
            minRSI = float(input(f"Minimum RSI (current {minRSI}): "))
            maxRSI = float(input(f"Maximum RSI (current {maxRSI}): "))

        useSMA20_50 = input("Use SMA 20/50 crossover? (y/n): ").lower() == "y"

        useADX = input("Use ADX? (y/n): ").lower() == "y"
        if useADX:
            setADX = float(input(f"ADX threshold (current {setADX}): "))

        useMomentum = input("Use Momentum? (y/n): ").lower() == "y"
        if useMomentum:
            minMomentum = float(input(f"Minimum momentum (current {minMomentum}): "))

        use3MG = input("Use 3-month growth? (y/n): ").lower() == "y"
        if use3MG:
            setMin3Month_Growth = float(input(f"Minimum 3-month growth (current {setMin3Month_Growth}): "))
            setMed3Month_Growth = float(input(f"Medium 3-month growth (current {setMed3Month_Growth}): "))
            setHigh3Month_Growth = float(input(f"High 3-month growth (current {setHigh3Month_Growth}): "))

        use6MG = input("Use 6-month growth? (y/n): ").lower() == "y"
        if use6MG:
            setMin6MG = float(input(f"Minimum 6-month growth (current {setMin6MG}): "))
            setMed6MG = float(input(f"Medium 6-month growth (current {setMed6MG}): "))
            setHigh6MG = float(input(f"High 6-month growth (current {setHigh6MG}): "))

        use1YG = input("Use 1-Year growth? (y/n): ").lower() == "y"
        if use1YG:
            setMin1yr = float(input(f"Minimum 1-Year Growth (current {setMin1yr}): "))
            setMed1yr = float(input(f"Medium 1-Year Growth (current {setMed1yr}): "))
            setHigh1yr = float(input(f"High 1-Year Growth (current {setHigh1yr}): "))

        useVol = input("Factor In Volatility? (y/n): ").lower() == "y"
        if useVol:
            minVol = float(input(f"Minimum Volatility (current {minVol}): "))
            maxVol = float(input(f"Maximum Volatility (current {maxVol}): "))

        useEMA = input("Use EMA? (y/n): ").lower() == "y"
        if useEMA:
            minEMA = float(input(f"Minimum EMA (current {minEMA}): "))

        useMACD = input("Use MACD? (y/n): ").lower() == "y"


        useSO = input("Use Stochastic Oscillator (y/n): ").lower() == "y"
        if useSO:
            minSO = float(input(f"Minimum Stochastic Value (0-100) (current {minSO}): "))
            maxSO = float(input(f"Maximum Stochastic Value (0-100) (current {maxSO}): "))


        print("\n✅ Parameters updated!\n")
        print("Enter 1 To Return To The Main Menu")
        userInput = input("Selection: ")

        while userInput != "1":
            userInput = input("Please Enter Your Choice: ")




    if userInput == "8":
        setStopLoss = -0.07
        setTakeProfit = 0.13
        useRSISell = True
        maxRSISell = 70
        useSMASell = True
        useMomentumeSell = True
        minMomentumSell = 30
        useADXSell = True
        minADXSell = 20

        print("\n--- Adjust Scoring Parameters ---\n")

        setStopLoss = float(input(f"Set Stop Loss (current {setStopLoss}): "))
        setTakeProfit = float(input(f"Set Take Profit (current {setStopLoss}): "))

        useRSISell = input("Use RSI as a Selling Indicator? (y/n): ").lower() == "y"
        if useRSISell:
            maxRSISell = float(input(f"Maximum RSI (current {maxRSISell}): "))

        useSMASell = input("Use SMA as a Selling Indicator? (y/n): ").lower() == "y"

        useMomentumeSell = input("Use Momentum as a Selling Indicator? (y/n): ").lower() == "y"
        if useMomentumeSell:
            minMomentumSell = float(input(f"Minimum Momentum (current {minMomentumSell}): "))

        useADXSell = input("Use ADX as a Selling Indicator? (y/n): ").lower() == "y"
        if useADXSell:
            minADXSell = float(input(f"Minimum ADX (current {minADXSell}): "))






    if userInput == "9":
        chosenStock = input("What Stock Would You Like To Analyze? ").upper()

    try:
        # Pull historical data
        df = yf.download(chosenStock, period="1y", interval="1d")  # 1 year to cover all growth windows
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()

        # === Indicators ===
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        ema_20 = ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1]

        macd_line = ta.trend.MACD(close).macd().iloc[-1]
        signal_line = ta.trend.MACD(close).macd_signal().iloc[-1]

        stoch = ta.momentum.StochasticOscillator(
            high=high,
            low=low,
            close=close,
            window=14,
            smooth_window=3
        ).stoch().iloc[-1]

        momentum = ta.momentum.ROCIndicator(close, window=10).roc().iloc[-1]

        growth_3mo = (close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]
        growth_6mo = (close.iloc[-1] - close.iloc[-126]) / close.iloc[-126]
        growth_1yr = (close.iloc[-1] - close.iloc[-220]) / close.iloc[-220]

        volatility = close.pct_change().rolling(20).std().iloc[-1]

        adx = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]

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




















