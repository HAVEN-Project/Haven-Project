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

    def compute_score (self):
        #Calculates score out of 
        score = 0
        #SMA Trend
        if(self.data.Close[-1] > self.sma20[-1]): score +=10
        if(self.data.Close[-1] > self.sma50[-1]): score +=10

        #SMA Cross
        if (self.sma20[-1]> self.sma50[-1]): score +=10

        #EMA Trend
        if (self.ema12[-1] > self.ema26[-1]): score +=10

        #RSI
        if ((self.rsi[-1] >= 40) & (self.rsi[-1] <= 70)): score += 10
        if ((self.rsi[-1] >= 50) & (self.rsi[-1] <= 60)): score += 10
        if (self.rsi[-1] > 80): score -5

        #ADX
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