import os
import re
import ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
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
def getSignal(data1,data2,data3,tf,st):
	high=data1[st:st+tf]
	low=data2[st:st+tf]
	close=data3[st:st+tf]
	startPrice=close[0]
	thPrice=startPrice*1.07
	tlPrice=startPrice*0.93	
	
	totl = 0
	toth = 0
	#print(tlPrice, thPrice)
	
	for i in range(len(high)):
		if high[i] > thPrice:
			toth+=1
		if low[i] < tlPrice:
			totl+=1

	
	if toth>totl:
		return np.array([1,0])#buy
	elif toth<totl:
		return np.array([0,1])#sell
	else:
		return np.array([0,0])#hodl

def getSignal2(data1,data2,data3,tf,st):
	high=data1[st:st+tf]
	low=data2[st:st+tf]
	close=data3[st:st+tf]
	startPrice=close[0]
	thPrice=startPrice*1.02
	tlPrice=startPrice*0.98

	#print(tlPrice, thPrice)
	new = []
	for i in range(len(high)):
		new.append([high[i],low[i]])
	for i in range(len(new)):
		if new[i][1] < tlPrice:
			return np.array([0,1])#sell
		if new[i][0] > thPrice:
			#print(new[i][0],thPrice)
			return np.array([1,0])#buy
		
	return np.array([0,0])#hodl
	
def normalize(inData):
	std=np.std(inData)
	mean=np.mean(inData)
	out=[]
	for i in range(len(inData)):
		t = ((inData[i]-mean)/std)
		out.append(t)
	return np.array(out)

	
#Data to aggregate
base = 'priceData/'
year = '2017'
pair='BTC-USD'
month = '1'
totalMonth = 12
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



csTime = 4
startTime = 144
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
inData=[[],[],[],[],[],[],[],[],[]]
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
inData[7] =np.array(list(ta.trend.mass_index(high, low, n=9*csTime, n2=25*csTime)))
#TRIX
inData[8] = np.array(list(ta.trend.trix(close, n=15*csTime)))


high = np.array(list(high))#used foy Y data
low = np.array(list(low))#used foy Y data
close = np.array(list(close))#used foy Y data

#trim all data to length of longest set, which it Slow EMA N
data=[[],[],[],[],[],[],[],[],[]]
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

patt=[0,5,10,15]

for i in range(startMin,len(data[0]),50):
	currX = []
	for j in data:
		tmpx = normalize(j[i-startMin:i])
		for k in patt:
			currX.append(tmpx[startMin-k-1])
	#print(currX)
	currY=getSignal(high,low,close,720,i)
	xData.append(np.array(currX))
	yData.append(np.array(currY))

xData=np.array(xData)
yData=np.array(yData)
t1x=[];t1y=[];t2x=[];t2y=[];t3x=[];t3y=[];
for i in range(len(xData)):
	if yData[i][0] == 1:
		t1x.append(xData[i])
		#t1y.append([1])
		t1y.append(yData[i])
	elif yData[i][1] == 1:
		t2x.append(xData[i])
		#t2y.append([0])
		t2y.append(yData[i])
	else:
		t3x.append(xData[i])
		#t3y.append([0.5])
		t3y.append(yData[i])
		
		
t = [len(t1x),len(t2x),len(t3x)]
print(t)
tm = min(t)-1
print(tm)

t1x,t1y = shuffleLists(t1x,t1y)
t2x,t2y = shuffleLists(t2x,t2y)
t3x,t3y = shuffleLists(t3x,t3y)

newXData = []
newYData = []

for i in range(tm):
	newXData.append(t1x[i])
	newXData.append(t2x[i])
	newXData.append(t3x[i])
	
	newYData.append(t1y[i])
	newYData.append(t2y[i])
	newYData.append(t3y[i])

newXData = np.array(newXData)
newYData = np.array(newYData)

newXData,newYData = shuffleLists(newXData,newYData)

testSize=int(len(newXData)/20)

trainX = newXData[testSize:]
trainY = newYData[testSize:]

testX  = newXData[:testSize]
testY  = newYData[:testSize]

print()
print(len(testY))
print(len(testX))
print()
print(len(trainX))
print(len(trainY))
print()
################################################
#Keras time!!
################################################

#define 
model = Sequential()
model.add(Dense(32, input_dim=36, activation='relu'))

#model.add(Dropout(0.25))
#model.add(Dense(10, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(2, activation='sigmoid'))


sgd = optimizers.adam(lr=0.001)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(trainX,trainY,epochs=100,batch_size=25)


scores=model.evaluate(testX,testY)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
