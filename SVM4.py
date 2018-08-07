import numpy as np
import pandas as pd
import ta
from matplotlib import style
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import functions

# This is so that when I print things to the console they display at the width of my monitor
style.use("ggplot")
console_width = 320
pd.set_option('display.width', console_width)
np.set_printoptions(linewidth=console_width)

"""
This makes use of the new data that we pulled from Kaggle, this data set is much more expansive though
    part of it is unusable, as the data set begins in 2012 where the maximum price fluctuation is less
    than $1,000. Even still it produces much more usable data than our previour data set that we scrapped
    from exchanges     
"""

# Declares empty list to load all of the data into
tdata = [[], [], [], [], [], []]

# Stands for candle stick size, in this case I am pulling data in 5 minute intervals
csSize = 5

# This is the file that contains all of the data we need
tdata =np.array(functions.aggregateData('btc_data.csv', csSize))

# Transposes the whole data set so that it's columns represent open, close, etc
train = np.array(tdata).T
# Seperates the initial part of the data that we don't want as mentioned above
train = train[200000:]

# Because I am training my SVM to predict over an hour or so I make sure the whole data set
# is divisible by 12 so that all of my arrays are the same size when I got to split them up.
# Note: Pulling data in 5 minute chunks, so 5 * 12 = 1 Hou
if len(train) % 12 != 0:
    chop = len(train) % 12
    train = train[:-chop]

# Split up the whole data set into the values we want to feed into our TA signals
train_tme = pd.Series(train[:, 0])
train_open = pd.Series(train[:, 1])
train_high = pd.Series(train[:, 2])
train_low = pd.Series(train[:, 3])
train_close = pd.Series(train[:, 4])
train_volume = pd.Series(train[:, 5])


#########################################################
####               TA Signals                        ####
#########################################################
rsi_short = ta.momentum.rsi(train_close, n=9)
tsi_short = ta.momentum.tsi(train_close, r=12, s=9)

rsi_medium = ta.momentum.rsi(train_close, n=12)
tsi_medium = ta.momentum.tsi(train_close, r=15, s=9)

rsi_medium2 = ta.momentum.rsi(train_close, n=15)
tsi_medium2 = ta.momentum.tsi(train_close, r=18, s=12)

rsi_long = ta.momentum.rsi(train_close, n=24)
tsi_long = ta.momentum.tsi(train_close, r=27, s=18)

rsi_exlong = ta.momentum.rsi(train_close, n=30)
tsi_exlong = ta.momentum.tsi(train_close, r=35, s=25)

mfi = ta.momentum.money_flow_index(train_high, train_low, train_close, train_volume, n=14)
macdSig = ta.trend.macd_signal(train_close, n_fast=12, n_slow=26, n_sign=9)

bband_high = ta.volatility.bollinger_hband_indicator(train_close, n=20)
bband_low = ta.volatility.bollinger_lband_indicator(train_close, n=20)


#########################################################
####                Data Handling                    ####
#########################################################


# Ensure all input array's are the same size
# The reason all arrays start from 48 is to ensure all TA signals have
# been calculated to avoid NAN or missing values

train_close = train_close[48:]
rsi_s = rsi_short[48:]
tsi_s = tsi_short[48:]

rsi_m = rsi_medium[48:]
tsi_m = tsi_medium[48:]

rsi_m2 = rsi_medium2[48:]
tsi_m2 = tsi_medium2[48:]

rsi_l = rsi_long[48:]
tsi_l = tsi_long[48:]

rsi_el = rsi_exlong[48:]
tsi_el = tsi_exlong[48:]

mfi = mfi[48:]
macdSig = macdSig[48:]
bband_h = bband_high[48:]
bband_l = bband_low[48:]


# Double check data for Nan values
"""
print(np.argwhere(np.isnan(train_close)))

print(np.argwhere(np.isnan(rsi_s)))
print(np.argwhere(np.isnan(rsi_m)))
print(np.argwhere(np.isnan(rsi_m2)))
print(np.argwhere(np.isnan(rsi_l)))
print(np.argwhere(np.isnan(rsi_el)))

print(np.argwhere(np.isnan(tsi_s)))
print(np.argwhere(np.isnan(tsi_m)))
print(np.argwhere(np.isnan(tsi_m2)))
print(np.argwhere(np.isnan(tsi_l)))
print(np.argwhere(np.isnan(tsi_el)))


print(np.argwhere(np.isnan(mfi)))
print(np.argwhere(np.isnan(macdSig)))
print(np.argwhere(np.isnan(bband_high)))
print(np.argwhere(np.isnan(bband_low)))
"""


# Separate price to calculate the expected y values for training
price = functions.split(train_close)
price = np.stack((price[0], price[1], price[2], price[3], price[4], price[5],
                  price[6], price[7], price[8], price[9], price[10], price[11]), axis=-1)


# Generate the X data set for training the SVM, this data set splits the list of TA values into rows
# representing an hour to fit with the rest of the model, ensuring that values are accounted for.
X = functions.splitAndCompress_Big(rsi_s, rsi_m, rsi_m2, rsi_l, rsi_el,
                                             tsi_s, tsi_m, tsi_m2, tsi_l, tsi_el,
                                             mfi, bband_l, bband_h)

# Get the length of X to make y the same length
size = len(X)
# Create empty y training set. This will array will serve as both our training and testing data
y = np.zeros(size)

# Double check that both array's are the same size.
print("The shape of X:", X.shape)
print("The shape of y:", y.shape)

# Generate y data for training and testing
for i in range(1, size):

    # This stores the price from the beginning of the last hour
    currClose = price.item((i - 1, 0))
    # This takes the price at the beginning of the next hour
    futureClose = price.item((i, 0))
    # Calculates the difference to generate the appropriate signal
    diff = (futureClose - currClose) / currClose

    # Checks to see if the difference is greater than half a percentage
    # This generates a buy signal
    if diff > 0.005:
        y[i - 1] = 1

    # This generates a sell signal
    elif diff < -0.005:
        y[i - 1] = -1

    # This generates a do nothing signal
    # The reason this was included is because a lot of market movement is
    # un-tradable and I want my SVM to be able to recognize sideways movement
    else:
        y[i - 1] = 0


## Signal Processing
##############################################################
# This section of code is helpful for optimization and debugging

# generate empty lists to tally up how many of what kind of signals are generated
# each list is paired so that the values of X are stored with their corresponding y values
t_side_x = []; t_side_y = []; t_buy_x = []; t_buy_y = []; t_sell_x = []; t_sell_y = []

# Tallies up each signal
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

# Converts them to np arrays
t_side_x = np.array(t_side_x)
t_side_y = np.array(t_side_y)
t_buy_x =  np.array(t_buy_x)
t_buy_y =  np.array(t_buy_y)
t_sell_x = np.array(t_sell_x)
t_sell_y = np.array(t_sell_y)

buys = len(t_buy_y)
sells = len(t_sell_y)
sideways = len(t_side_y)

# After storing how many buys, sells, and sideways movement signals they are returned to
# to represent the whole data set.
X = np.concatenate((t_side_x, t_buy_x, t_sell_x), axis=0)
y = np.concatenate((t_side_y, t_buy_y, t_sell_y), axis=0)

print(t_sell_x.shape)
print(t_sell_y.shape)

# Double check everything made it back into the arrays
print(X.shape)
print(y.shape)

print("Buys:", buys, "Sells:", sells, "Sideways:", sideways)

# Double check that no NAN values appeared or missing values
print(np.argwhere(np.isnan(X)))

# Shuffle everything so that we are ready to train!
X, y = functions.shuffleLists(X, y)

# One last check that no NAN values appeared from the shuffle
print(np.argwhere(np.isnan(X)))

#########################################################
####                 SVM Time                        ####
#########################################################

# Split the data up into test and train
setSize = len(X)
trainSize = .85 * setSize

print(trainSize)

train_X = X[:34171]
train_y = y[:34171]

test_X = X[34171:]
test_y = y[34171:]


clf1 = SGDClassifier(loss='modified_huber', penalty='elasticnet', max_iter=1000, n_jobs=-1,
                    l1_ratio=.15,
                    learning_rate='optimal', alpha=0.01, class_weight={-1: .34, 0: 0.30, 1: 0.36})

clf1.fit(train_X, train_y)

# Call the prediction method to compare accuracy
results1_y = clf1.predict(test_X)

# Compare some of the signals generated
print(results1_y[:50])
print(test_y[:50])

acc1 = accuracy_score(test_y, results1_y)
# acc2 = accuracy_score(test_y, results2_y)
# acc3 = accuracy_score(test_y, results3_y)


print("Accuracy of model 1:", acc1)
# print("Accuracy of model 2:", acc2)
# print("Accuracy of model 3:", acc3)

# joblib.dump(clf1, "SVM_Model_svm4.pkl")

mat = confusion_matrix(test_y, results1_y)

sns.heatmap(mat, square=True, annot=True, cbar=False) #, cmap='YlGnBu', flag, YlGnBu, jet
plt.xlabel('predicted value')
plt.ylabel('true value')

plt.show()
