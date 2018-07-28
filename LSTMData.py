import os
import re
import ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def aggregateData(fn, eStep):
	nd = re.compile(r'[^\d.]+')
	f = open(fn)
	stepSize = eStep * 60
	currData = f.readline().split(',')
	currData = f.readline().split(',')
	for i in range(len(currData)):
		currData[i] = float(nd.sub('', currData[i]))
	tme = int(currData[0])
	outData = [[], [], [], [], [], []]
	for line in f:
		t = line.split(',')
		for i in range(len(t)):
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
		if t[0] >= tme + stepSize:
			for i in range(6):
				outData[i].append(currData[i])
			tme += stepSize
			currData = [0] * 6
			currData[0] = tme
			currData[1] = t[1]
			currData[2] = t[2]
			currData[3] = t[3]
			currData[4] = t[4]
			currData[5] = t[5]
		else:
			if t[2] > currData[2]:
				currData[2] = t[2]

			if t[3] < currData[3] and t[3] != 0:
				currData[3] = t[3]

			currData[4] = t[4]

			currData[5] += t[5]
	for i in range(6):
		outData[i].append(currData[i])
	return outData


def aggregate(fn, eStep):
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
	outData = []
	for line in f:
		t = line.split(',')
		for i in range(len(t)):
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

		if t[0] >= tme + stepSize:
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
			if currData[1] > t[1]:
				currData[1] = t[1]

			if currData[2] < t[2]:
				currData[2] = t[2]

			currData[4] = t[4]

			currData[5] += t[5]
	outData.append(currData)
	return outData

outFile = 'trainData/data.data'
#base = 'priceData/'
#year = '2016'
#pair = 'BTC-USD'
#month = '1'
#totalMonth = 24
#tm is the output candlestick
#tm = 5
#tdata = [[], [], [], [], [], []]

#for i in range(totalMonth):
#
#	tmp = base + year + '/' + pair + month + 'min.data'
#	currData = aggregate(tmp, tm)
#
#	print(tmp)
#
#	for i in range(int(len(currData))):
#		for j in range(6):
#			tdata[j].append(currData[i][j])
#
#	month = str(int(month) + 1)
#	if int(month) > 12:
#		month = str(int(month) - 12)
#		year = str(int(year) + 1)
#

#data = tdata

data=np.array(aggregateData('btcData.csv',5))

#signals to collect for NN
#keltner uppand and lower
keltN = 15
#RSI
rsiN=14
#EMA fast and EMA slow
emaFN = 12
emaSN = 26
#macd slow and fast N
macdFN = 12
macdSN = 26
#vortex Indicator


ichin1 = 9
ichin2 = 26
ichin3 = 52

mN = max([keltN,rsiN,emaFN,emaSN,macdFN,macdSN,ichin1,ichin2,ichin3])
print(mN)
inData=[[],[],[],[],[],[],[],[],[]]
print(len(data[4]))
close = pd.Series(data[4])
high  = pd.Series(data[2])
low   = pd.Series(data[1])


inData[0] = np.array(list(ta.momentum.rsi(close,n=rsiN)))/100

inData[1] = np.array(list(ta.momentum.tsi(close,r=25,s=13)))/100

inData[2] = np.array(list(ta.momentum.rsi(close, n=rsiN + 5))) / 100

inData[3] = np.array(list(ta.momentum.tsi(close, r=28, s=15))) / 100

inData[4] = np.array(list(ta.trend.ema_fast(close,n_fast=emaFN)))

inData[5] = np.array(list(ta.trend.ema_slow(close,n_slow=emaSN)))

inData[6] = np.array(list(ta.volatility.keltner_channel_hband(high, low, close, n=keltN)))

inData[7] = np.array(list(ta.volatility.keltner_channel_lband(high, low, close, n=keltN)))

inData[8] = np.array(close)

for i in range(len(inData)):
	inData[i] = inData[i][mN:]
print(min(close))

#plt.plot(inData[8])
#plt.show()



cl = inData[8]
for i in range(len(inData)):
	inData[i] = inData[i][mN:]

for i in range(len(inData[4])):
	inData[4][i] = (cl[i] - inData[4][i]) / inData[4][i]

for i in range(len(inData[5])):
	inData[5][i] = (cl[i] - inData[5][i]) / inData[5][i]

for i in range(len(inData[6])):
	inData[6][i] = (cl[i] - inData[6][i]) / inData[6][i]

for i in range(len(inData[7])):
	inData[7][i] = (cl[i] - inData[7][i]) / inData[7][i]

for i in inData:
	print(min(i), max(i))
f=open(outFile, 'w')
for i in range(200000,len(inData[0])-60000):

	curr = ''
	for j in range(len(inData) - 1):
		curr = curr + str(inData[j][i]) + ','
	curr = curr + str(inData[-1][i]) + '\n'
	f.write(curr)
f.close()

#Mass Index
#inData[7] =np.array(list(ta.trend.mass_index(high, low, n=9*csTime, n2=25*csTime)))
#TRIX
#inData[8] = np.array(list(ta.trend.trix(close, n=15*csTime)))
