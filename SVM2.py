import numpy as np
import ta
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm
import functions
import pandas as pd
import pickle
from sklearn import preprocessing

style.use("ggplot")


tdata = [[], [], [], [], [], []]

tdata =np.array(functions.aggregateData('btc_data.csv', 5))

tdata = np.array(tdata).T

train = tdata[200006:-60000]
test = tdata[-60000:]

print(train.shape)
print(test.shape)

train_tme = pd.Series(train[:, 0])
train_open = pd.Series(train[:, 1])
train_high = pd.Series(train[:, 2])
train_low = pd.Series(train[:, 3])
train_close = pd.Series(train[:, 4])
train_volume = pd.Series(train[:, 5])

print(train_close.shape)


# rsi = ta.momentum.rsi(train_close, n=14)
# tsi = ta.momentum.tsi(train_close, r=25, s=13)
# mfi = ta.momentum.money_flow_index(train_high, train_low, train_close, train_volume, n=14)

# easeOf = ta.volume.ease_of_movement(train_high, train_low, train_close, train_volume, n=20)

# avgDir = ta.trend.adx(train_high, train_low, train_close, n=14)
# macd = ta.trend.macd_diff(train_close, n_fast=12, n_slow=26, n_sign=9)
# macdSig = ta.trend.macd_signal(train_close, n_fast=12, n_slow=26, n_sign=9)




# vortex_pos = ta.trend.vortex_indicator_pos(train_high, train_low, train_close, n=14)
# vortex_neg = ta.trend.vortex_indicator_neg(train_high, train_low, train_close, n=14)

print(train_open.shape)
print(train_close.shape)


##############################################################
##                         SVM TIME
##############################################################


## Data Handling
##############################################################


# Build the data to import to SVM by chunking up the data into sets for classifying the next p amount of time based on
# on the past h

# I want a batch size of a half an hour, which is broken into 6 columns per factor



# This value keeps track of the index of batch so that the data lines up




# Splits the training data of price into 12 colunms representing an hour of data per row
train_close_0 = train_close[::12]
train_close_1 = train_close[1::12]
train_close_2 = train_close[2::12]
train_close_3 = train_close[3::12]
train_close_4 = train_close[4::12]
train_close_5 = train_close[5::12]

train_close_6 = train_close[6::12]
train_close_7 = train_close[7::12]
train_close_8 = train_close[8::12]
train_close_9 = train_close[9::12]
train_close_10 = train_close[10::12]
train_close_11 = train_close[11::12]


train_close_Total = np.stack((train_close_0, train_close_1, train_close_2,
                              train_close_3, train_close_4, train_close_5,
                              train_close_6, train_close_7, train_close_8,
                              train_close_9, train_close_10, train_close_11), axis=-1)

print(train_close_Total.shape)
print(train_close_Total[:5])


size = len(train_close_Total)
X = train_close_Total
y = np.zeros((size))

trades = []
trades = np.array(trades)

for i in range(1, size):

    currClose = X.item((i-1, 0))
    futureClose = X.item((i, 11))
    # if i < 10:
    #    futureClose = X.item((i, 30, 1))

    #print(currClose)
    diff = (currClose - futureClose) / currClose
    if (diff > .02 or diff < -.02):
        trades.append(diff)
    if diff > 0.02:
        y[i-1] = 1
    elif diff < -0.02:
        y[i-1] = -1
    else:
        y[i-1] = 0

print(y.shape)

clf = svm.SVC(gamma=.0001, C=100)
clf.fit(X, y)

print(test.shape)
print(max(trades))
print(min(trades))
print(len(trades))






"""
# train_open_t = train_open.values.reshape([1, size, 1])
# train_volume_t = train_volume.values.reshape([1, size, 1])


# rsi_t = rsi.values.reshape([1, size, 1])
# tsi_t = tsi.values.reshape([1, size, 1])
# mfi_t = mfi.values.reshape([1, size, 1])
# easeOf_t = easeOf.values.reshape([1, size, 1])
# avgDir_t = avgDir.values.reshape([1, size, 1])
# macd_t = macd.values.reshape([1, size, 1])
# macdSig_t = macdSig.values.reshape([1, size, 1])


#  Range starts at 2 because the first frame is devoted to starting the calculations for
#  all of the moving averages

for i in range(1, num_Batches+1, 1):

    # Populates X with with i sets of matrices size 1, ind , 1
    # X[i-1:i, :1] = train_open_t[:1, batch_index - batch_Size:batch_index, :1]


    # X[i-1:i, 1:2] = train_close_t[:1, batch_index - batch_Size:batch_index, :1]
    # X[i-1:i, 2:3] = train_volume_t[:1, batch_index - batch_Size:batch_index, :1]

    # X[i-1:i, ::, 3:4] = rsi_t[:1, batch_index - batch_Size:batch_index, :1]
    # X[i-1:i, ::, 4:5] = tsi_t[:1, batch_index - batch_Size:batch_index, :1]
    # X[i-1:i, ::, 5:6] = easeOf_t[:1, batch_index - batch_Size:batch_index, :1]
    # X[i-1:i, ::, 6:7] = avgDir_t[:1, batch_index - batch_Size:batch_index, :1]
    # X[i-1:i, ::, 7:8] = macd_t[:1, batch_index - batch_Size:batch_index, :1]
    # X[i-1:i, ::, 8:9] = macdSig_t[:1, batch_index - batch_Size:batch_index, :1]

    batch_index += 60


#print(X[:1, -5:, :])
print((X.shape))


# print(y[:10,::,::])
# print(X[:5, :10, :3])




# test = np.arange(50).reshape((1, 50, 1))
# test2 = test[:1, 25:50, :1]
# X[:1, :25, :1] = test[:1, :25, :1]
# X[:1, 25:50, 1:2] = test2
# print(X[:1, ::, :3])

print(y.shape)



fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(bottom=0.2)
ax1.plot(train_close[:], color='tab:green')

# ax2.plot(tsi[14:], color='tab:red')
# ax2.plot(vortex_pos[14:], color='tab:red')
# ax2.plot(vortex_neg[14:], color='tab:red')
# ax2.plot(tsi_mid[startTime:], color='tab:blue')
# ax2.plot(tsi_short[startTime:], color='tab:green')
# ax2.plot(tsi_EMA[startTime:], color='tab:orange')
# ax2.plot(tsi_EMA_Bollinger_High[startTime:], color='tab:brown')
# ax2.plot(tsi_EMA_Bollinger_Low[startTime:], color='tab:green')

# ax2.axhline(33, color='darkgoldenrod')
# ax2.axhline(0, color='darkgoldenrod')
# ax2.axhline(-24, color='darkgoldenrod')

# plt.show()


"""
