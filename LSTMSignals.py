import os
import re
import ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM
from keras import optimizers

def shuffleLists(l1,l2):
	rng=np.random.get_state()
	np.random.shuffle(l1)
	np.random.set_state(rng)
	np.random.shuffle(l2)
	return l1,l2
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
def getSignal3(data1, tf, st):
	close = data1[st:st + tf]
	sp = close[0]
	tl = sp * 0.97
	th = sp * 1.03
	for i in close:
		if i > th:
			return np.array([1, 0, 0])
		elif i < tl:
			return np.array([0, 0, 1])
	return np.array([0, 1, 0])


def getSignal4(data1, data2, tf, st):
	close = data1[st:st + tf]
	ema = data2[st:st + tf]

	if ema[0] > close[0]:
		for i in range(len(close)):
			if ema[0] < close[0]:
				return np.array([1, 0, 0])

		return np.array([0, 1, 0])
	else:
		for i in range(len(close)):
			if ema[0] > close[0]:
				return np.array([0, 0, 1])
		return np.array([0, 1, 0])


def normalize(inData):
	std=np.std(inData)
	mean=np.mean(inData)
	out=[]
	for i in range(len(inData)):
		t = ((inData[i]-mean)/std)
		out.append(t)
	#print(std, mean)
	return np.array(out)

	
#Data to aggregate
base = 'priceData/'
year = '2016'
pair='BTC-USD'
month = '1'
totalMonth = 24
tdata = [[],[],[],[],[],[]]

for i in range(totalMonth):
	
	tmp = base+year + '/' + pair + month + 'min.data'
	currData = aggregate(tmp,1,1)
	
	print(tmp)
	
	for i in range(int(len(currData))):
		for j in range(6):
			tdata[j].append(currData[i][j])
			
	month = str(int(month)+1)
	if int(month) > 12:
		month = str(int(month)-12)
		year = str(int(year)+1)



csTime = 5
startTime = 288*2
startMin = csTime*startTime

#tme = pd.Series(data[0])
#low = pd.Series(data[1])
#high = pd.Series(data[2])
#open = pd.Series(data[3])
#close = pd.Series(data[4])
#volume = pd.Series(data[5])

data = tdata
#signals to collect for NN	
#keltner uppand and lower
keltN = 15 * csTime
#RSI
rsiN=14 * csTime
#EMA fast and EMA slow
emaFN = 12 * csTime
emaSN = 26 * csTime
#macd slow and fast N
macdFN = 12 * csTime
macdSN = 26 * csTime
#vortex Indicator


ichin1 = 9 * csTime
ichin2 = 26 * csTime
ichin3 = 52 * csTime

mN = max([keltN,rsiN,emaFN,emaSN,macdFN,macdSN,ichin1,ichin2,ichin3])
print(mN)
inData=[[],[],[],[],[],[],[]]
print(len(data[4]))
close = pd.Series(data[4])
high  = pd.Series(data[2])
low   = pd.Series(data[1])


#close = inData[0]
inData[0] = np.array(list(data[4]))
#print(inData[0])
#rsi  = inData[1]
inData[1] = np.array(list(ta.momentum.rsi(close,n=rsiN)))
#emaF = inData[2]
inData[2] = np.array(list(ta.trend.ema_fast(close,n_fast=emaFN)))
#emaS = inData[3]
inData[3] = np.array(list(ta.trend.ema_slow(close,n_slow=emaSN)))
#keltH = inData[4]
inData[4] = np.array(list(ta.volatility.keltner_channel_hband(high, low, close, n=keltN)))
#keltL = inData[5]
inData[5] = np.array(list(ta.volatility.keltner_channel_lband(high, low, close, n=keltN)))
#MACD
inData[6] = np.array(list(ta.trend.macd(close, n_fast=macdFN,n_slow=macdSN)))
#Mass Index
#inData[7] =np.array(list(ta.trend.mass_index(high, low, n=9*csTime, n2=25*csTime)))
#TRIX
#inData[8] = np.array(list(ta.trend.trix(close, n=15*csTime)))


high = np.array(list(high))#used foy Y data
low = np.array(list(low))#used foy Y data
close = np.array(list(close))#used foy Y data

#trim all data to length of longest set, which it Slow EMA N
data=[[],[],[],[],[],[],[]]
for i in range(len(data)):
	data[i]=inData[i][mN:]
#for i in range(len(data)):
#	data[i]=normalize(inData[i])
	
#input layer of NN
#6 features with 4 timestamps each, for a total of 24 input nodes
#(close[0,15,30,60],RSI[0,15,30,60],EMAF[0,15,30,60],EMAS[0,15,30,60],KeltH[0,15,30,60],KeltH[0,15,30,60])

#output layer is simple 2 node ouput
#(1,0) buy signal
#(0,1) sell signal
#(0,0) neutral

xData = []
yData = []


lstm1Len = 24

for i in range(startMin,len(data[0]),lstm1Len):
	currX = [];
	tmpData = []
	for j in data:
		tmpData.append(normalize(j[i-startMin:i])[-lstm1Len:])

	tLen = len(tmpData[0])
	for j in range(tLen-lstm1Len-1,tLen-1):
		currInnerX = []
		for k in range(len(tmpData)):
			currInnerX.append(tmpData[k][j])
		currX.append(currInnerX)
	xData.append(currX)
	yData.append(getSignal3(inData[0],144,i))

xData=np.array(xData)
yData=np.array(yData)
#for i in range(len(xData)):
#	xData[i] = np.rot90(xData[i])
#yData = normalize(yData)
print(xData.shape)
print(yData.shape)
t1x=[];t1y=[];t2x=[];t2y=[];t3x=[];t3y=[];
for i in range(len(yData)):
	#print(yData[i])
	#print(yData[i])
	if yData[i][0] == 1:
		t1x.append(xData[i])
		t1y.append(1)
		#t1y.append(yData[i])
		#t1y.append([1,-1])
	elif yData[i][2] == 1:
		t2x.append(xData[i])
		t2y.append(0)
		#t2y.append(yData[i])
		#t2y.append([-1,1])
	else:
		t3x.append(xData[i])
		#t3y.append([0])
		t3y.append(yData[i])
		#t3y.append([0,0])
		
		
t = [len(t1x),len(t2x)]#,len(t3x)]
print(t)
tm = min(t)-1

t1x,t1y = shuffleLists(t1x,t1y)
t2x,t2y = shuffleLists(t2x,t2y)
t3x,t3y = shuffleLists(t3x,t3y)

newXData = []
newYData = []

for i in range(tm):
	newXData.append(t1x[i])
	newYData.append(t1y[i])
	
	newXData.append(t2x[i])
	newYData.append(t2y[i])
	
	#newXData.append(t3x[i])
	#newYData.append(t3y[i])

newXData = np.array(newXData)
newYData = np.array(newYData)

newXData,newYData = shuffleLists(newXData,newYData)

testSize=int(len(newXData)/20)

trainX = newXData[testSize:]
trainY = newYData[testSize:]

testX  = newXData[:testSize]
testY  = newYData[:testSize]

################################################
#Keras time!!
################################################

#define 
model = Sequential()
model.add(LSTM(
		input_shape=(lstm1Len,7),
		units=100,
		return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
		units=200,
		return_sequences=False))
model.add(Dropout(0.2))
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))


sgd = optimizers.SGD(lr=0.0001)
model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])
print(trainX.shape)
model.fit(trainX,trainY,epochs=40,batch_size=50)


scores=model.evaluate(testX,testY)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
