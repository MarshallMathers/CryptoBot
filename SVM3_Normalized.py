import numpy as np
from numpy.random import RandomState
import ta
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm
import functions
import pandas as pd
from sklearn import preprocessing

style.use("ggplot")
console_width = 320
pd.set_option('display.width', console_width)
np.set_printoptions(linewidth=console_width)


def shuffleLists(l1, l2):
    rng = RandomState(1234)
    rng.shuffle(l1)
    rng.shuffle(l2)
    return l1, l2


base = 'priceData/'
year = '2016'
pair = 'BTC-USD'
month = '2'
totalMonth = 18

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

train = np.array(tdata).T
train = train[:-4]

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
macd = ta.trend.macd(train_close, n_fast=12, n_slow=26)

# Signals for maybe later
#
# easeOf = ta.volume.ease_of_movement(train_high, train_low, train_close, train_volume, n=20)
# avgDir = ta.trend.adx(train_high, train_low, train_close, n=14)
# macd = ta.trend.macd_diff(train_close, n_fast=12, n_slow=26, n_sign=9)


# vortex_pos = ta.trend.vortex_indicator_pos(train_high, train_low, train_close, n=14)
# vortex_neg = ta.trend.vortex_indicator_neg(train_high, train_low, train_close, n=14)

print(train_open.shape)
print(train.shape)






##############################################################
##                         SVM TIME
##############################################################


## Data Handling
##############################################################

train_close = train_close[48:]
rsi = rsi[48:]
tsi = tsi[48:]
mfi = mfi[48:]
macd = macd[48:]


train_Total_noPrice = functions.splitAndCompress_noPrice(rsi, tsi, mfi, macd)
train_Total = functions.splitAndCompress(train_close, rsi, tsi, mfi, macd)

X = train_Total
Xnp = train_Total_noPrice
Xnp = preprocessing.normalize(Xnp, norm='l2')

size = len(X)
y = np.zeros(size)

print("The shape of X:", X.shape)
print("The shape of y:", y.shape)

# Generate y data for training
for i in range(1, size):

    currClose = X.item((i - 1, 0))
    futureClose = X.item((i, 30))
    diff = (futureClose - currClose) / currClose

    if diff > 0.01:
        y[i - 1] = 1
    elif diff < -0.01:
        y[i - 1] = -1
    else:
        y[i - 1] = 0

t_side_x = [];
t_side_y = [];
t_buy_x = [];
t_buy_y = [];
t_sell_x = [];
t_sell_y = [];

for i in range(len(X)):
    if y[i] == 0:
        t_side_x.append(Xnp[i])
        # t_side_x.append(X[i])
        t_side_y.append(0)
    elif y[i] == 1:
        t_buy_x.append(Xnp[i])
        #t_buy_x.append(X[i])
        t_buy_y.append(1)
    else:
        t_sell_x.append(Xnp[i])
        #t_sell_x.append(X[i])
        t_sell_y.append(-1)

t_side_x = np.array(t_side_x)
t_side_y = np.array(t_side_y)
t_buy_x =  np.array(t_buy_x)
t_buy_y =  np.array(t_buy_y)
t_sell_x = np.array(t_sell_x)
t_sell_y = np.array(t_sell_y)

t_side_x = t_side_x[:770]
t_side_y = t_side_y[:770]
t_buy_x =  t_buy_x[:770]
t_buy_y =  t_buy_y[:770]
t_sell_x = t_sell_x[:770]
t_sell_y = t_sell_y[:770]




buys = len(t_buy_y)
sells = len(t_sell_y)
sideways = len(t_side_y)


X = np.concatenate((t_side_x, t_buy_x, t_sell_x), axis=0)
y = np.concatenate((t_side_y, t_buy_y, t_sell_y), axis=0)

print(t_sell_x.shape)
print(t_sell_y.shape)


print(X.shape)
print(y.shape)

print("Buys:", buys, "Sells:", sells, "Sideways:", sideways)



X, y = shuffleLists(X, y)

train_X = X[:1700]
train_y = y[:1700]

test_X = X[1700:]
test_y = y[1700:]

clf1 = svm.SVC(gamma=.0001, C=100, kernel="rbf", cache_size=1000)
clf1.fit(train_X, train_y)

# clf2 = svm.SVC(gamma=.001, C=100, kernel="poly")
# clf2.fit(train_X, train_y)

# clf3 = svm.SVC(gamma=.0001, C=100, kernel="sigmoid")
# clf3.fit(train_X, train_y)

results1_y = clf1.predict(test_X)
# results2_y = clf2.predict(test_X)
# results3_y = clf3.predict(test_X)

# print(results_y[:50])
# print(test_y[:50])




from sklearn.metrics import accuracy_score

acc1 = accuracy_score(test_y, results1_y)
# acc2 = accuracy_score(test_y, results2_y)
# acc3 = accuracy_score(test_y, results3_y)


print("Accuracy of model 1:", acc1)
# print("Accuracy of model 2:", acc2)
# print("Accuracy of model 3:", acc3)

