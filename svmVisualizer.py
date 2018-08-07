import re
import ta
import functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
This file should be pretty easy to read given the documentation of the other more complex files.
    This file served as the testing grounds for visualizing the data in different ways.
    I actually am not using the TA a whole lot, I mostly used this file to plot the graphs of price
    to get a better understanding of what I was trying to do.
"""

base = 'priceData/'
year = '2016'
pair = 'BTC-USD'
month = '2'
totalMonth = 1

# Candle Stick size in minutes
csSize = 5

tdata = [[], [], [], [], [], []]

for i in range(totalMonth):

    tmp = base + year + '/' + pair + month + 'min.data'
    currData = functions.aggregate(tmp, 1, csSize)

    print(tmp)

    for i in range(int(len(currData))):
        for j in range(6):
            tdata[j].append(currData[i][j])

    month = str(int(month) + 1)
    if int(month) > 12:
        month = str(int(month) - 12)
        year = str(int(year) + 1)


cs1_Hour = 12
cs1_Day = cs1_Hour * 24

startTime = cs1_Day * 30
startMin = cs1_Hour * startTime

data = tdata

# Split up the whole data set into the values we want to feed into our TA signals
low = pd.Series(data[1])
high = pd.Series(data[2])
open = pd.Series(data[3])
close = pd.Series(data[4])
volume = pd.Series(data[5])

#########################################################
####               TA Signals                        ####
#########################################################

tsi_long = ta.momentum.tsi(close, r=30, s=20)
tsi_short = ta.momentum.tsi(close, r=18, s=15)

tsi_EMA = ta.trend.ema_slow(tsi_long, n_slow=100)
tsi_EMA_Bollinger_High = ta.volatility.bollinger_hband(tsi_EMA, n=50, ndev=3)
tsi_EMA_Bollinger_Low = ta.volatility.bollinger_lband(tsi_EMA, n=50, ndev=3)



fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(bottom=0.2)
ax1.plot(close[:500], color='tab:green')

ax2.plot(tsi_long[30:500], color='tab:red')
ax2.plot(tsi_short[18:500], color='tab:green')
# ax2.plot(tsi_EMA[startTime:], color='tab:orange')
# ax2.plot(tsi_EMA_Bollinger_High[startTime:], color='tab:brown')
# ax2.plot(tsi_EMA_Bollinger_Low[startTime:], color='tab:green')
ax2.xaxis.set_ticks(np.arange(0, 500, 5))
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




