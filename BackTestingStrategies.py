from backtesting import Strategy
import pandas as pd
import pandas_ta as ta
import numpy as np
from backtesting.lib import crossover

class MyStrategy(Strategy):
    def init(self):
        # getting data
        open = pd.Series(self.data.Open)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        volume = pd.Series(self.data.Volume)
        
        #indicators
        self.sma20 = self.I(lambda x: ta.sma(x, length=20), close)
        self.sma50 = self.I(lambda x: ta.sma(x, length=50), close)
        self.ema12 = self.I(lambda x: ta.ema(x, length=12), close)
        self.ema26 = self.I(lambda x: ta.ema(x, length=26), close)
        self.rsi = self.I(lambda x: ta.rsi(x, length=14), close)
        self.adx = self.I(lambda x, y, z: ta.adx(x, y, z, length=12), high, low, close)

        #Score
        self.score_indicator = self.I(lambda: np.full_like(self.data.Close, np.nan))

    def compute_score_buy (self):
        #Calculates score out of 
        score = 0 # This should be a % from 0-100 (ex: 73% match)
        #SMA 10 dip Trend
        sma10dip = 0.032 #This is 3.2% by default
        if "{sma10 dips {sma10dip}":
            score += {"excel grabs sma10 weight out of 100"}

        #RSI
        if ((self.rsi[-1] >= 40) & (self.rsi[-1] <= 70)):
            score += {"excel grabs RSI weight out of 100"}

        #ATR (Buy signal)
        if 3 < {atr} < 4
            score += {"excel grabs ATR weight out of 100"}

        if stochastic crosses 20 from the bottom:
            score += {"excel grabs stoch weight out of 100"}

        if 3mogrowth is greater than 25%:
            score += {"excel grabs 3mogrowth weight out of 100"}

        if 1 yrgrwth is greater than 2%:
            score += {"excel grabs 1yrgrwth weight out of 100"}
        return score

    def compute_score_sell(self):
        # Calculates score out of
        score = 0  # This should be a % from 0-100 (ex: 73% match)
        if self.position and price < self.atrts[-1]:
            return 100 #returns 100% match to sell
        if current profit > 15%:
            return 100 #returns 100% match to sell
        # RSI
        if RSI > 70:
            score += {"excel grabs RSI weight out of 100"}

        if stochastic crosses 80 from the top:
            score += {"excel grabs stoch weight out of 100"}






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

class Xatrts(Strategy):
    def init(self):
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)

        # Compute ATR trailing stop
        self.atrts = self.I(lambda h, l, c: ta.atrts(h, l, c, length=14, k=3.5), high, low, close)

        self.adx = self.I(lambda x, y, z: ta.adx(x, y, z, length=12), high, low, close)

    def next(self):
        price = self.data.Close[-1]

        # If we have no open position and price breaks ABOVE trailing stop → BUY
        if not self.position and price > self.atrts[-1] and self.data.Close[-2] <= self.atrts[-2]:
            self.buy()

        # If we are in a long position and price breaks BELOW trailing stop → SELL (exit)
        elif self.position and price < self.atrts[-1]:
            self.position.close()

class Phase1(Strategy):
    def init(self):
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)

        #ATR trailing stop loss
        self.atrts = self.I(lambda h, l, c: ta.atrts(h, l, c, length=14, k=3.5), high, low, close)

        #SMA 10,20
        self.sma10 = self.I(lambda x: ta.sma(x, length=10), close)
        self.sma20 = self.I(lambda x: ta.sma(x, length=20), close)

    def next(self):
        if not self.position and self.data.Close[-1] > self.atrts[-1] and self.data.Close[-2] <= self.atrts[-2] and self.sma10 > self.sma20:
            self.buy()

        elif self.position and self.data.Close[-1] < self.atrts[-1]:
            self.position.close()