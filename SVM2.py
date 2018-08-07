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


tdata = [[], [], [], [], [], []]

tdata =np.array(functions.aggregateData('btc_data.csv', 5))

train = np.array(tdata).T
train = train[200000:]

if len(train) % 12 != 0:
    chop = len(train) % 12
    train = train[:-chop]

train_tme = pd.Series(train[:, 0])
train_open = pd.Series(train[:, 1])
train_high = pd.Series(train[:, 2])
train_low = pd.Series(train[:, 3])
train_close = pd.Series(train[:, 4])
train_volume = pd.Series(train[:, 5])




# Signals being used
rsi_short = ta.momentum.rsi(train_close, n=9)
tsi_short = ta.momentum.tsi(train_close, r=14, s=9)
rsi_long = ta.momentum.rsi(train_close, n=14)
tsi_long = ta.momentum.tsi(train_close, r=25, s=13)

mfi = ta.momentum.money_flow_index(train_high, train_low, train_close, train_volume, n=14)
macdSig = ta.trend.macd_signal(train_close, n_fast=12, n_slow=26, n_sign=9)

bband_high = ta.volatility.bollinger_hband_indicator(train_close, n=20)
bband_low = ta.volatility.bollinger_lband_indicator(train_close, n=20)

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

train_close = train_close[48:]
rsi_s = rsi_short[48:]
rsi_l = rsi_long[48:]
tsi_s = tsi_short[48:]
tsi_l = tsi_long[48:]
mfi = mfi[48:]
macdSig = macdSig[48:]
bband_high = bband_high[48:]
bband_low = bband_low[48:]

print(np.argwhere(np.isnan(train_close)))
print(np.argwhere(np.isnan(rsi_l)))
print(np.argwhere(np.isnan(rsi_s)))
print(np.argwhere(np.isnan(tsi_l)))
print(np.argwhere(np.isnan(tsi_s)))
print(np.argwhere(np.isnan(mfi)))
print(np.argwhere(np.isnan(macdSig)))
print(np.argwhere(np.isnan(bband_high)))
print(np.argwhere(np.isnan(bband_low)))


price = functions.split(train_close)
price = np.stack((price[0], price[1], price[2], price[3], price[4], price[5],
                  price[6], price[7], price[8], price[9], price[10], price[11]), axis=-1)


train_Total = functions.splitAndCompress_noPrice(rsi_s, tsi_s, rsi_l, tsi_l, mfi, bband_low, bband_high)


X = train_Total

size = len(X)
y = np.zeros(size)

print("The shape of X:", X.shape)
print("The shape of y:", y.shape)

# Generate y data for training
# Generate y data for training
for i in range(1, size):

    currClose = price.item((i - 1, 0))
    futureClose = price.item((i, 0))
    diff = (futureClose - currClose) / currClose

    if diff > 0.005:
        y[i - 1] = 1
    elif diff < -0.005:
        y[i - 1] = -1
    else:
        y[i - 1] = 0

t_side_x = []; t_side_y = []; t_buy_x = []; t_buy_y = []; t_sell_x = []; t_sell_y = []

for i in range(len(X)):
    if y[i] == 0:
        t_side_x.append(X[i])
        t_side_y.append(0)
    elif y[i] == 1:
        t_buy_x.append(X[i])
        t_buy_y.append(1)
    else:
        t_sell_x.append(X[i])
        t_sell_y.append(-1)

t_side_x = np.array(t_side_x)
t_side_y = np.array(t_side_y)
t_buy_x =  np.array(t_buy_x)
t_buy_y =  np.array(t_buy_y)
t_sell_x = np.array(t_sell_x)
t_sell_y = np.array(t_sell_y)

# t_side_x = t_side_x[:2900]
# t_side_y = t_side_y[:2900]
# t_buy_x =  t_buy_x[:2900]
# t_buy_y =  t_buy_y[:2900]
# t_sell_x = t_sell_x[:2900]
# t_sell_y = t_sell_y[:2900]




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



from sklearn.linear_model import SGDClassifier

print(np.argwhere(np.isnan(X)))

X, y = functions.shuffleLists(X, y)

print(np.argwhere(np.isnan(X)))

train_X = X[:30000]
train_y = y[:30000]

test_X = X[30000:]
test_y = y[30000:]


# clf1 = svm.SVC(gamma=.000001, C=100, kernel="rbf", cache_size=1000, class_weight={0: 0.15})
# clf1.fit(train_X, train_y)

clf1 = SGDClassifier(loss='modified_huber', penalty='elasticnet', max_iter=1000, n_jobs=-1,
                     learning_rate='optimal', alpha=0.0001, class_weight={-1: .507, 0: 0.05, 1: 0.475})
clf1.fit(train_X, train_y)

# clf3 = svm.SVC(gamma=.0001, C=100, kernel="sigmoid")
# clf3.fit(train_X, train_y)

results1_y = clf1.predict(test_X)
# results2_y = clf2.predict(test_X)
# results3_y = clf3.predict(test_X)

print(results1_y[:50])
print(test_y[:50])




from sklearn.metrics import accuracy_score

acc1 = accuracy_score(test_y, results1_y)
# acc2 = accuracy_score(test_y, results2_y)
# acc3 = accuracy_score(test_y, results3_y)


print("Accuracy of model 1:", acc1)
# print("Accuracy of model 2:", acc2)
# print("Accuracy of model 3:", acc3)

joblib.dump(clf1, "SVM_Model.pkl")


import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

mat = confusion_matrix(test_y, results1_y)

sns.heatmap(mat, square=True, annot=True, cbar=False) #, cmap='YlGnBu', flag, YlGnBu, jet
plt.xlabel('predicted value')
plt.ylabel('true value')

plt.show()

