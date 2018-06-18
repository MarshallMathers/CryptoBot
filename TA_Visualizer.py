import os
import re
import ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MONDAY, DateFormatter, DayLocator, WeekdayLocator

#import matplotlib.finance
#from matplotlib.finance import candlestick2_ohlc as candlestick

def aggregate(fn, sStep, eStep):
    nd = re.compile(r'[^\d.]+')
    f = open(fn)
    stepSize = eStep * 60
    currData = f.readline().split(',')

    for i in range(len(currData)):
        if '.' not in currData[i]:
            currData[i] = int(nd.sub('', currData[i]))
        else:
            currData[i] = float(nd.sub('', currData[i]))

    tme = int(currData[0])
    # low = t[1]
    # high = t[2]
    # open = t[3]
    # close = t[4]
    # vol = t[5]
    outData = []
    for line in f:
        t = line.split(',')

        for i in range(len(t)):

            # This is formatting stuff for ensuring periods and blank spaces are handled
            if '.' not in t[i]:
                tmp = nd.sub('', t[i])
                if tmp == '':
                    t[i] = 0
                else:
                    t[i] = int(tmp)
            else:
                tmp = nd.sub('', t[i])
                if tmp == '':
                    t[i] = 0
                else:
                    t[i] = float(tmp)

        # Checks to see if current frame t is a new candle stick
        if t[0] > tme + stepSize:
            # if it is a new candle stick, append old candle stick data to outData
            # initialize new candle stick data set with current frame.
            outData.append(currData)
            tme += stepSize
            currData = [0] * 6
            currData[0] = tme
            currData[1] = t[1]
            currData[2] = t[2]
            currData[3] = t[3]
            currData[4] = t[4]
            currData[5] = t[5]
        else:
            # currData is current candle stick
            # Checks to see if at current frame t, the price is lower than
            # the recorded low in currData.
            if currData[1] > t[1]:
                currData[1] = t[1]

            # Checks to see if at current frame t, the price is higher than
            # the recorded high in currData.
            if currData[2] < t[2]:
                currData[2] = t[2]

            # Close data
            currData[4] = t[4]

            # Volume for candle stick
            currData[5] += t[5]

    outData.append(currData)
    return outData


# def plot(data):


base = 'priceData/'
year = '2017'
pair = 'ETH-USD'
month = '4'
totalMonth = 5

# date1 = "2017-1-1"
# date2 = "2017-9-30"

# mondays = WeekdayLocator(MONDAY)
# alldays = DayLocator()
# weekFormatter = DateFormatter('%b %d')
# dayFormatter = DateFormatter('%d')

# Candle Stick size in minutes
csSize = 60

tdata = [[], [], [], [], [], []]

for i in range(totalMonth):

    tmp = base + year + '/' + pair + month + 'min.data'
    currData = aggregate(tmp, 1, csSize)

    print(tmp)

    for i in range(int(len(currData))):
        for j in range(6):
            tdata[j].append(currData[i][j])

    month = str(int(month) + 1)
    if int(month) > 12:
        month = str(int(month) - 12)
        year = str(int(year) + 1)


cs1_Hour = 1
cs1_Day = cs1_Hour * 24

startTime = cs1_Day * 30
startMin = cs1_Hour * startTime

data = tdata

tme = pd.Series(data[0])
low = pd.Series(data[1])
high = pd.Series(data[2])
open = pd.Series(data[3])
close = pd.Series(data[4])
volume = pd.Series(data[5])

"""
# mk3trader settings
tsi_long = ta.momentum.tsi(close, r=42, s=30)
tsi_short = ta.momentum.tsi(close, r=18, s=15)

tsi_EMA = ta.trend.ema_slow(tsi_long, n_slow=100)
tsi_EMA_Bollinger_High = ta.volatility.bollinger_hband(tsi_EMA, n=75, ndev=3)
tsi_EMA_Bollinger_Low = ta.volatility.bollinger_lband(tsi_EMA, n=75, ndev=3)
"""

tsi_long = ta.momentum.tsi(close, r=30, s=40)
tsi_short = ta.momentum.tsi(close, r=18, s=15)

tsi_EMA = ta.trend.ema_slow(tsi_long, n_slow=100)
tsi_EMA_Bollinger_High = ta.volatility.bollinger_hband(tsi_EMA, n=50, ndev=3)
tsi_EMA_Bollinger_Low = ta.volatility.bollinger_lband(tsi_EMA, n=50, ndev=3)


# tsi_short = ta.momentum.tsi(close, r=15, s=12)

# tsi_mid = ta.trend.ema_slow(tsi_short, n_slow=25)



fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(bottom=0.2)
ax1.plot(close[startTime:], color='tab:green')

ax2.plot(tsi_long[startTime:], color='tab:red')
# ax2.plot(tsi_mid[startTime:], color='tab:blue')
#ax2.plot(tsi_short[startTime:], color='tab:green')
#ax2.plot(tsi_EMA[startTime:], color='tab:orange')
ax2.plot(tsi_EMA_Bollinger_High[startTime:], color='tab:brown')
ax2.plot(tsi_EMA_Bollinger_Low[startTime:], color='tab:green')

ax2.axhline(33, color='darkgoldenrod')
ax2.axhline(0, color='darkgoldenrod')
ax2.axhline(-24, color='darkgoldenrod')




# ax.xaxis.set_major_locator(mondays)
# ax.xaxis.set_minor_locator(alldays)
# ax.xaxis.set_major_formatter(weekFormatter)
# ax.xaxis_date()
# ax.autoscale_view()

#plt.subplot(212)

plt.show()



"""
tmp = aggregate('priceData/2018/BTC-USD2min.data', 1, 5)
tmp2 = [[], [], [], [], [], []]

for i in range(int(len(tmp) / 20)):
    for j in range(6):
        tmp2[j].append(tmp[i][j])
tme = pd.Series(tmp2[0])
low = pd.Series(tmp2[1])
high = pd.Series(tmp2[2])
open = pd.Series(tmp2[3])
close = pd.Series(tmp2[4])
volume = pd.Series(tmp2[5])

"""















#kh = ta.volatility.keltner_channel_hband(high, low, close)
#kl = ta.volatility.keltner_channel_lband(high, low, close)
#macd = ta.trend.macd(close)
#macd2 = ta.trend.macd_diff(close)
#ichi = ta.trend.ichimoku_a(high, low)
#emv = ta.volume.ease_of_movement(high, low, close, volume)
#trix = ta.trend.trix(close)

#VI1 = ta.trend.vortex_indicator_pos(high, low, close, n=26)
#VI2 = ta.trend.vortex_indicator_neg(high, low, close, n=26)

# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.set_ylabel('VI',color=color)
# ax1.plot(VI1,color=color)
# ax1.plot(VI2,color='tab:green')
# ax2 = ax1.twinx()
# color='tab:blue'
# ax2.set_ylabel('Price',color=color)
# ax2.plot(close,color=color)
# fig.tight_layout()

# plt.plot(ema2)
# plt.plot(close)



