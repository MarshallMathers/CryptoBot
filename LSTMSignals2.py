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

data = [[],[],[],[],[],[],[],[],[]]
f = open('trainData/data.data')
for l in f:
	t = l.split(',')
	for i in range(len(t)):
		data[i].append(float(t[i]))
data = np.array(data)
xData = []
yData = []


lstm1Len = 24

for i in range(lstm1Len,len(data[0]),lstm1Len):
	currX = [];
	for j in range(i-lstm1Len,i):
		currInnerX = []
		for k in range(len(data)-1):
			currInnerX.append(data[k][j])
		currX.append(currInnerX)
	xData.append(currX)
	yData.append(getSignal3(data[8],144,i))

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
		input_shape=(lstm1Len,8),
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


sgd = optimizers.SGD(lr=0.001)
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
