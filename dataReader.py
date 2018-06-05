import os
import re
import ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def aggregate(fn,sStep,eStep):
    nd = re.compile(r'[^\d.]+')
    f = open(fn)
    stepSize = eStep*60
    currData = f.readline().split(',')
    for i in range(len(currData)):
        if '.' not in currData[i]:
            currData[i] = int(nd.sub('',currData[i]))
        else:
            currData[i] = float(nd.sub('',currData[i]))

    tme = int(currData[0])
    #low = t[1]
    #high = t[2]
    #open = t[3]
    #close = t[4]
    #vol = t[5]
    outData = []
    for line in f:
        t = line.split(',')
        for i in range(len(t)):
            if '.' not in t[i]:
                t[i] = int(nd.sub('',t[i]))
            else:
                t[i] = float(nd.sub('',t[i]))

        if t[0] > tme + stepSize:
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
            if currData[1] >  t[1]:
                currData[1] = t[1]

            if currData[2] < t[2]:
                currData[2] = t[2]

            currData[4] = t[4]

            currData[5] += t[5]
    outData.append(currData)
    return outData

#for filename in os.listdir(os.getcwd() + '/priceData/2017'):
#	f = open('priceData/2017/' + filename)
#	start = 0
#	counter = 0
#	for line in f:
#		start +=60
#		if int(line.split(',')[0]) != start:
#			#print('aaah')
#			counter +=1
#	print(filename + ': ' +str(counter))
tmp = aggregate('priceData/2018/BTC-USD2min.data',1,5)
tmp2 = [[],[],[],[],[],[]]

for i in range(int(len(tmp)/20)):
    for j in range(6):
        tmp2[j].append(tmp[i][j])
tme = pd.Series(tmp2[0])
low = pd.Series(tmp2[1])
high = pd.Series(tmp2[2])
open = pd.Series(tmp2[3])
close = pd.Series(tmp2[4])
volume = pd.Series(tmp2[5])

kh = ta.volatility.keltner_channel_hband(high,low,close)
kl = ta.volatility.keltner_channel_lband(high,low,close)
macd = ta.trend.macd(close)
macd2 = ta.trend.macd_diff(close)
ichi = ta.trend.ichimoku_a(high,low)
emv = ta.volume.ease_of_movement(high,low,close,volume)
trix = ta.trend.trix(close)

VI1 = ta.trend.vortex_indicator_pos(high,low,close,n=26)
VI2 = ta.trend.vortex_indicator_neg(high,low,close,n=26)

#fig, ax1 = plt.subplots()
#color = 'tab:red'
#ax1.set_ylabel('VI',color=color)
#ax1.plot(VI1,color=color)
#ax1.plot(VI2,color='tab:green')
#ax2 = ax1.twinx()
#color='tab:blue'
#ax2.set_ylabel('Price',color=color)
#ax2.plot(close,color=color)
#fig.tight_layout()

plt.figure(1)
plt.subplot(211)
plt.plot(VI2,color='tab:green')
plt.plot(VI1,color='tab:red')
plt.subplot(212)
plt.plot(close)

plt.show()


#plt.plot(ema2)
#plt.plot(close)
plt.show()