import numpy as np
import ta
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm
import functions
import pandas as pd
from sklearn.externals import joblib
from sklearn import preprocessing

style.use("ggplot")
console_width = 320
pd.set_option('display.width', console_width)
np.set_printoptions(linewidth=console_width)


def shuffleLists(l1, l2):
    rng = np.random.get_state()
    np.random.shuffle(l1)
    np.random.set_state(rng)
    np.random.shuffle(l2)
    return l1, l2

tdata = [[], [], [], [], [], []]

tdata =np.array(functions.aggregateData('btc_data.csv', 5))

train = np.array(tdata).T
train = train[200000:-2]

# ta.train = tdata[200006:-60000]
# train = tdata[200006:-60000]
# test = tdata[-60000:]

train_tme = pd.Series(train[:, 0])
train_open = pd.Series(train[:, 1])
train_high = pd.Series(train[:, 2])
train_low = pd.Series(train[:, 3])
train_close = pd.Series(train[:, 4])
train_volume = pd.Series(train[:, 5])

# Signals being used
rsi = ta.momentum.rsi(train_close, n=14)
tsi = ta.momentum.tsi(train_close, r=25, s=13)
mfi = ta.momentum.money_flow_index(train_high, train_low, train_close, train_volume, n=14)
macdSig = ta.trend.macd_signal(train_close, n_fast=12, n_slow=26, n_sign=9)

# Signals for maybe later
#
# easeOf = ta.volume.ease_of_movement(train_high, train_low, train_close, train_volume, n=20)
# avgDir = ta.trend.adx(train_high, train_low, train_close, n=14)
# macd = ta.trend.macd_diff(train_close, n_fast=12, n_slow=26, n_sign=9)


# vortex_pos = ta.trend.vortex_indicator_pos(train_high, train_low, train_close, n=14)
# vortex_neg = ta.trend.vortex_indicator_neg(train_high, train_low, train_close, n=14)

# print(train_close[:50])
# print(rsi[300:306])
# print(tsi[300:306])
# print(mfi[300:306])
# print(macdSig[300:306])

##############################################################
##                         SVM TIME
##############################################################


## Data Handling
##############################################################


# Splits the training data of price into 12 colunms representing an hour of data per row

train_close = train_close[100:]
rsi = rsi[100:]
tsi = tsi[100:]
mfi = mfi[100:]
macdSig = macdSig[100:]

train_Total = functions.splitAndCompress(train_close, rsi, tsi, mfi, macdSig)


X = train_Total

size = len(X)
y = np.zeros(size)

print("The shape of X:", X.shape)
print("The shape of y:", y.shape)

# Generate y data for training
for i in range(1, size):

    currClose = X.item((i-1, 0))
    futureClose = X.item((i, 30))
    diff = (futureClose - currClose) / currClose
    if diff > 0.01:
        y[i-1] = 1
    elif diff < -0.01:
        y[i-1] = -1
    else:
        y[i-1] = 0

X, y = shuffleLists(X, y)

print(X.shape)
print(y.shape)

train_X = X[:3600]
train_y = y[:3600]

test_X = X[36000:]
test_y = y[36000:]


print(np.argwhere(np.isnan(X)))


clf = svm.SVC(gamma=.0001, C=100)
clf.fit(train_X, train_y)

results_y = clf.predict(test_X)

print(results_y[:50])
print(test_y[:50])

from sklearn.metrics import accuracy_score

acc = accuracy_score(test_y, results_y)
print(acc)

"""
# joblib.dump(clf, "SVM_Model.pkl")

"""