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
					tmp = nd.sub('',t[i])
					if tmp == '':
						t[i]=0
					else:
						t[i] = int(tmp)
				else:
					tmp = nd.sub('',t[i])
					if tmp == '':
						t[i]=0
					else:
						t[i] = float(tmp)
		
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

#data: pandas series of 6 tracked datapoints
#tf: timeframe to identify signals
#st: start time
#
def getSignal(data,tf,st):
	emaN = 26
	atrN = 14
	bbN  = 8
	f, (ax1,ax2,ax3,ax4) = plt.subplots(4)
	if st>=emaN:
		emaCloseData=pd.Series(data[4][st-emaN:st+tf])
		ema = ta.trend.ema_slow(emaCloseData,n_slow=emaN)
		startPrice = emaCloseData[atrN-1]
		print(len(emaCloseData[emaN:]))
		print(len(emaCloseData))
		ax1.plot(emaCloseData,color='tab:blue')
		ax1.plot(ema,color='tab:green')
		
	if st>=atrN:
		atrCloseData=pd.Series(data[4][st-atrN:st+tf])
		atrHighData=pd.Series(data[2][st-atrN:st+tf])
		atrLowData=pd.Series(data[1][st-atrN:st+tf])
		atr = ta.volatility.average_true_range(atrHighData,atrLowData,atrCloseData,n=atrN)
		
		ax2.plot(atr,color='tab:red')
	
	if st>=bbN:
		bbCloseData=pd.Series(data[4][st-bbN:st+tf])
		bbHighData=pd.Series(data[2][st-bbN:st+tf])
		bbLowData=pd.Series(data[1][st-bbN:st+tf])
		bbh = ta.volatility.keltner_channel_hband(bbHighData,bbLowData,bbCloseData,n=bbN)
		bbl = ta.volatility.keltner_channel_lband(bbHighData,bbLowData,bbCloseData,n=bbN)
		bbc = ta.volatility.keltner_channel_central(bbHighData,bbLowData,bbCloseData,n=bbN)
		
		ax3.plot(bbCloseData[bbN:],color='tab:blue')
		ax3.plot(bbh,color='tab:red')
		ax3.plot(bbl,color='tab:red')
		
		ax4.plot(bbc,color='tab:blue')
		ax4.plot(bbh,color='tab:red')
		ax4.plot(bbl,color='tab:red')
	
	plt.show()
	

base = 'priceData/'
year = '2017'
pair='BTC-USD'
month = '1'
totalMonth = 7
tdata = [[],[],[],[],[],[]]

for i in range(totalMonth):
	tmp = base+year + '/' + pair + month + 'min.data'
	currData = aggregate(tmp,1,5)
	print(tmp)
	
	for i in range(int(len(currData))):
		for j in range(6):
			tdata[j].append(currData[i][j])
			
	month = str(int(month)+1)
	if int(month) > 12:
		month = str(int(month)-12)
		year = str(int(year)+1)


#tme = pd.Series(data[0])
#low = pd.Series(data[1])
#high = pd.Series(data[2])
#open = pd.Series(data[3])
#close = pd.Series(data[4])
#volume = pd.Series(data[5])
data = pd.Series(tdata)

for i in range(20):
	getSignal(data,100,1000*(i+1))
print(len(data[0]))


