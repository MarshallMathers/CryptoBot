import numpy as np
import ta
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm
import functions
import pandas as pd
from sklearn import preprocessing

style.use("ggplot")

# def split_data(year, pair, startingMonth, totalMonths):


base = 'priceData/'
year = '2016'
pair = 'BTC-USD'
month = '2'
totalMonth = 2



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


tdata = np.array(tdata).T

train = tdata[:15840]


print(len(train))

# test = tdata[578:]
print(tdata.shape)

train_tme = pd.Series(train[:, 0])
train_low = pd.Series(train[:, 1])
train_high = pd.Series(train[:, 2])
train_open = pd.Series(train[:, 3])
train_close = pd.Series(train[:, 4])
train_volume = pd.Series(train[:, 5])


rsi = ta.momentum.rsi(train_close, n=14)
tsi = ta.momentum.tsi(train_close, r=25, s=13)
mfi = ta.momentum.money_flow_index(train_high, train_low, train_close, train_volume, n=14)

easeOf = ta.volume.ease_of_movement(train_high, train_low, train_close, train_volume, n=20)

avgDir = ta.trend.adx(train_high, train_low, train_close, n=14)
macd = ta.trend.macd_diff(train_close, n_fast=12, n_slow=26, n_sign=9)
macdSig = ta.trend.macd_signal(train_close, n_fast=12, n_slow=26, n_sign=9)




# vortex_pos = ta.trend.vortex_indicator_pos(train_high, train_low, train_close, n=14)
# vortex_neg = ta.trend.vortex_indicator_neg(train_high, train_low, train_close, n=14)

print(train_open.shape)
print(train_close.shape)

"""
##############################################################
##                         SVM TIME
##############################################################


## Data Handling
##############################################################
"""

# Build the data to import to SVM by chunking up the data into sets for classifying the next p amount of time based on
# on the past h

batch_Size = 60
num_Batches = int(len(train)/batch_Size)


# This value keeps track of the index of batch so that the data lines up
batch_index = 60


size = len(train_close)
X = np.zeros((num_Batches, batch_Size, 9))
y = np.zeros((num_Batches, 1))

# Reshapes the datasets into 3D matracies for eazy conversion
train_close_t = train_close.values.reshape([1, size, 1])
train_open_t = train_open.values.reshape([1, size, 1])
train_volume_t = train_volume.values.reshape([1, size, 1])


rsi_t = rsi.values.reshape([1, size, 1])
tsi_t = tsi.values.reshape([1, size, 1])
mfi_t = mfi.values.reshape([1, size, 1])
easeOf_t = easeOf.values.reshape([1, size, 1])
avgDir_t = avgDir.values.reshape([1, size, 1])
macd_t = macd.values.reshape([1, size, 1])
macdSig_t = macdSig.values.reshape([1, size, 1])


#  Range starts at 2 because the first frame is devoted to starting the calculations for
#  all of the moving averages
for i in range(1, num_Batches+1, 1):

    # Populates X with with i sets of matrices size 1, ind , 1
    X[i-1:i, ::, :1] = train_open_t[:1, batch_index - batch_Size:batch_index, :1]
    X[i-1:i, ::, 1:2] = train_close_t[:1, batch_index - batch_Size:batch_index, :1]
    X[i-1:i, ::, 2:3] = train_volume_t[:1, batch_index - batch_Size:batch_index, :1]
    X[i-1:i, ::, 3:4] = rsi_t[:1, batch_index - batch_Size:batch_index, :1]
    X[i-1:i, ::, 4:5] = tsi_t[:1, batch_index - batch_Size:batch_index, :1]
    X[i-1:i, ::, 5:6] = easeOf_t[:1, batch_index - batch_Size:batch_index, :1]
    X[i-1:i, ::, 6:7] = avgDir_t[:1, batch_index - batch_Size:batch_index, :1]
    X[i-1:i, ::, 7:8] = macd_t[:1, batch_index - batch_Size:batch_index, :1]
    X[i-1:i, ::, 8:9] = macdSig_t[:1, batch_index - batch_Size:batch_index, :1]

    batch_index += 60


print(X[:1, -5:, :])
print((X.shape))


for i in range(1, num_Batches + 1):

    currClose = X.item((i-1, -1, 1))

    if i < 10:
        futureClose = X.item((i, 30, 1))

    #print(currClose)
    diff = (currClose - futureClose) / currClose
    #print(diff)
    if diff > 0.03:
        y[i-1:i, :] = 1

    else:
        y[i-1:i, :] = 0

# print(y[:10,::,::])
# print(X[:5, :10, :3])




# test = np.arange(50).reshape((1, 50, 1))
# test2 = test[:1, 25:50, :1]
# X[:1, :25, :1] = test[:1, :25, :1]
# X[:1, 25:50, 1:2] = test2
# print(X[:1, ::, :3])

print(y.shape)

clf = svm.SVC(gamma=.0001, C=100)
clf.fit(X, y)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(bottom=0.2)
ax1.plot(train_close[:], color='tab:green')

ax2.plot(tsi[14:], color='tab:red')
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