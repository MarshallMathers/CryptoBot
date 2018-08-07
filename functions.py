import re
import numpy as np
from numpy.random import RandomState

"""
This file serves as a helper file for my SVM's and TA based trader files

The first two aggregate data functions deal with importing two different data sets:
    
    aggregateData: deals with a more robust data set that was pulled from Kaggle and features
        a longer period of usable data
        
    aggregate: deals with our own scrapped data from exchange servers and due to limitations of
        historic data, features a smaller usable data set. We transitioned over to the larger data set
        about 2/3 into this project hence why only the most recent files feature aggregateData instead 
        of this function
        
The functions listed after this are as follows:
    
    split: takes any time series data and splits it into 12 columns representing an hour in each row.
        The function itself only returns a list of lists that is then used in the various splitAndCompress
        functions to assemble rows of data that hold an hour of 5 minute data points.
    
    splitAndCompress_Price: This was the first attempt at compiling features for an SVM, this function takes in
        price, rsi, tsi, mfi. I wasn't normalizing the data as I was testing out the functionality of this approach.
        I quickly realized that price was extremely hard to normalize and for the most part irrelevant to the 
        effectiveness of the algorithm.
    
    splitAndCompress_noPrice: This was my first function that started to gain traction, but I knew that I needed more
        features so I started to build out helper functions such as split and then signalCruncher as my features
        list started to get big.
    
    signalCruncher: This was my patch job for the fact that the splitAndCompress functions were a 
        lot of repeated code, signalCruncher takes in the same signal at 5 different time intervals and normalizes
        them as well as compresses them into a np.array 5 columns wide. 
    
    splitAndCompress_Big: Finally started to see accuracy over 50% after adding more features, this function totals
        156 features mostly from the added rsi and tsi time frames. This is also the first function to implement 
        signalCruncher.
    
    splitAndCompress_Massive: Final stage of the evolution, represents 226 features from the added time frame
        windows from the mfi 1, 2, and 3. The same goes for bollinger bands.
        
    shuffle: Pretty self explanatory but it takes in the X and y data set and shuffles them together so that the
        values remained paired.

"""

# TODO: I would like to work on compressing these functions into for loops
# but had trouble figuring out how to add columns via for loop.


# Handling larger dataset from Kaggle
def aggregateData(fn, eStep):
    nd = re.compile(r'[^\d.]+')
    f = open(fn)
    stepSize = eStep * 60
    currData = f.readline().split(',')
    currData = f.readline().split(',')
    for i in range(len(currData)):
        currData[i] = float(nd.sub('', currData[i]))
    tme = int(currData[0])
    outData = [[], [], [], [], [], []]
    for line in f:
        t = line.split(',')
        for i in range(len(t)):
            if '.' not in t[i]:
                tmp = nd.sub('', t[i])
                if tmp == '':
                    t[i] = 0
                else:
                    t[i] = int(tmp)
            else:
                tmp = nd.sub('', t[i])
                if tmp == '':
                    t[i] = 0
                else:
                    t[i] = float(tmp)
        if t[0] >= tme + stepSize:
            for i in range(6):
                outData[i].append(currData[i])
            tme += stepSize
            currData = [0] * 6
            currData[0] = tme
            currData[1] = t[1]
            currData[2] = t[2]
            currData[3] = t[3]
            currData[4] = t[4]
            currData[5] = t[5]
        else:
            if t[2] > currData[2]:
                currData[2] = t[2]

            if t[3] < currData[3] and t[3] != 0:
                currData[3] = t[3]

            currData[4] = t[4]

            currData[5] += t[5]
    for i in range(6):
        outData[i].append(currData[i])
    return outData


# Old aggregate function that deals with our scrapped.
def aggregate(fn, sStep, eStep):
    nd = re.compile(r'[^\d.]+')
    f = open(fn)
    stepSize = eStep * 60
    currData = f.readline().split(',')

    for i in range(len(currData)):
        if '.' not in currData[i]:
            currData[i] = int(nd.sub('', currData[i]))
        else:
            currData[i] = float(nd.sub('', currData[i]))

    tme = int(currData[0])
    # low = t[1]
    # high = t[2]
    # open = t[3]
    # close = t[4]
    # vol = t[5]
    outData = []
    for line in f:
        t = line.split(',')

        for i in range(len(t)):

            # This is formatting stuff for ensuring periods and blank spaces are handled
            if '.' not in t[i]:
                tmp = nd.sub('', t[i])
                if tmp == '':
                    t[i] = 0
                else:
                    t[i] = int(tmp)
            else:
                tmp = nd.sub('', t[i])
                if tmp == '':
                    t[i] = 0
                else:
                    t[i] = float(tmp)

        # Checks to see if current frame t is a new candle stick
        if t[0] > tme + stepSize:
            # if it is a new candle stick, append old candle stick data to outData
            # initialize new candle stick data set with current frame.
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
            # currData is current candle stick
            # Checks to see if at current frame t, the price is lower than
            # the recorded low in currData.
            if currData[1] > t[1]:
                currData[1] = t[1]

            # Checks to see if at current frame t, the price is higher than
            # the recorded high in currData.
            if currData[2] < t[2]:
                currData[2] = t[2]

            # Close data
            currData[4] = t[4]

            # Volume for candle stick
            currData[5] += t[5]

    outData.append(currData)
    return outData


# This function splits the data and helps cut down on excessive code.
def split(ta_signal):
    out = []
    for i in range(12):
        out.append(ta_signal[i::12])
    return out


# First version of split and compress, this handles price but was quickly abandoned due to price being
# impossible to normalize.
def splitAndCompress_Price(close, rsi, tsi, mfi):

    close_list = split(close)

    # Set rsi short to be between 0 and 1
    rsi = rsi/100
    rsi_list = split(rsi)

    # Set tsi long to be between 0 and 1
    tsi = tsi / 100
    tsi_list = split(tsi)

    # Set mfi to be between 0 and 1
    mfi = mfi / 100
    mfi_list = split(mfi)

    # This stacks all of the lists together so that it becomes one long row of data points
    #   which acts as the time frame the SVM learns on.
    train_Total = np.stack((close_list[0],  rsi_list[0],  tsi_list[0],  mfi_list[0],
                            close_list[1],  rsi_list[1],  tsi_list[1],  mfi_list[1],
                            close_list[2],  rsi_list[2],  tsi_list[2],  mfi_list[2],
                            close_list[3],  rsi_list[3],  tsi_list[3],  mfi_list[3],
                            close_list[4],  rsi_list[4],  tsi_list[4],  mfi_list[4],
                            close_list[5],  rsi_list[5],  tsi_list[5],  mfi_list[5],
                            close_list[6],  rsi_list[6],  tsi_list[6],  mfi_list[6],
                            close_list[7],  rsi_list[7],  tsi_list[7],  mfi_list[7],
                            close_list[8],  rsi_list[8],  tsi_list[8],  mfi_list[8],
                            close_list[9],  rsi_list[9],  tsi_list[9],  mfi_list[9],
                            close_list[10], rsi_list[10], tsi_list[10], mfi_list[10],
                            close_list[11], rsi_list[11], tsi_list[11], mfi_list[11]), axis=-1)

    return train_Total


# Second version of split and compress, doesn't take in price and normalizes the data.
def splitAndCompress_noPrice(rsi_s, tsi_s, rsi_l, tsi_l, mfi, bband_l, bband_h):

    # Set rsi short to be between 0 and 1
    rsi_s = rsi_s/100
    rsi_list_s = split(rsi_s)

    # Set tsi short to be between 0 and 1
    tsi_s = tsi_s/100
    tsi_list_s = split(tsi_s)

    # Set rsi long to be between 0 and 1
    rsi_l = rsi_l / 100
    rsi_list_l = split(rsi_l)

    # Set tsi long to be between 0 and 1
    tsi_l = tsi_l / 100
    tsi_list_l = split(tsi_l)

    # Set mfi to be between 0 and 1
    mfi = mfi / 100
    mfi_list = split(mfi)

    # Set stoch to be between 0 and 1
    bband_h_list = split(bband_h)
    bband_l_list = split(bband_l)

    rsi_tsi_s_l = np.stack((rsi_list_s[0],  tsi_list_s[0],  rsi_list_l[0],  tsi_list_l[0],
                            rsi_list_s[1],  tsi_list_s[1],  rsi_list_l[1],  tsi_list_l[1],
                            rsi_list_s[2],  tsi_list_s[2],  rsi_list_l[2],  tsi_list_l[2],
                            rsi_list_s[3],  tsi_list_s[3],  rsi_list_l[3],  tsi_list_l[3],
                            rsi_list_s[4],  tsi_list_s[4],  rsi_list_l[4],  tsi_list_l[4],
                            rsi_list_s[5],  tsi_list_s[5],  rsi_list_l[5],  tsi_list_l[5],
                            rsi_list_s[6],  tsi_list_s[6],  rsi_list_l[6],  tsi_list_l[6],
                            rsi_list_s[7],  tsi_list_s[7],  rsi_list_l[7],  tsi_list_l[7],
                            rsi_list_s[8],  tsi_list_s[8],  rsi_list_l[8],  tsi_list_l[8],
                            rsi_list_s[9],  tsi_list_s[9],  rsi_list_l[9],  tsi_list_l[9],
                            rsi_list_s[10], tsi_list_s[10], rsi_list_l[10], tsi_list_l[10],
                            rsi_list_s[11], tsi_list_s[11], rsi_list_l[11], tsi_list_l[11]), axis=-1)

    mfi_bband_high_low = np.stack((mfi_list[0],  bband_l_list[0],  bband_h_list[0],
                                   mfi_list[1],  bband_l_list[1],  bband_h_list[1],
                                   mfi_list[2],  bband_l_list[2],  bband_h_list[2],
                                   mfi_list[3],  bband_l_list[3],  bband_h_list[3],
                                   mfi_list[4],  bband_l_list[4],  bband_h_list[4],
                                   mfi_list[5],  bband_l_list[5],  bband_h_list[5],
                                   mfi_list[6],  bband_l_list[6],  bband_h_list[6],
                                   mfi_list[7],  bband_l_list[7],  bband_h_list[7],
                                   mfi_list[8],  bband_l_list[8],  bband_h_list[8],
                                   mfi_list[9],  bband_l_list[9],  bband_h_list[9],
                                   mfi_list[10], bband_l_list[10], bband_h_list[10],
                                   mfi_list[11], bband_l_list[11], bband_h_list[11]), axis=-1)

    sweetJeebus = np.concatenate((rsi_tsi_s_l, mfi_bband_high_low), axis=1)

    return sweetJeebus


# Helper function for V3 and forward, cuts down on function size
# In short it takes in 5 different time level signals and combines them into
# a data set that is 5 columns wide with each row representing a time frame of an hour.
def signalCruncher(signal_s, signal_m, signal_m2, signal_l, signal_el):

    # Set signal short to be between 0 and 1
    signal_s = signal_s / 100
    signal_list_s = split(signal_s)

    # Set signal medium to be between 0 and 1
    signal_m = signal_m / 100
    signal_list_m = split(signal_m)

    # Set signal medium2 to be between 0 and 1
    signal_m2 = signal_m2 / 100
    signal_list_m2 = split(signal_m2)

    # Set signal long to be between 0 and 1
    signal_l = signal_l / 100
    signal_list_l = split(signal_l)

    # Set signal extra long to be between 0 and 1
    signal_el = signal_el / 100
    signal_list_el = split(signal_el)

    signal_stack1 = np.stack((signal_list_s[0],  signal_list_m[0],  signal_list_m2[0],
                              signal_list_s[1],  signal_list_m[1],  signal_list_m2[1],
                              signal_list_s[2],  signal_list_m[2],  signal_list_m2[2],
                              signal_list_s[3],  signal_list_m[3],  signal_list_m2[3],
                              signal_list_s[4],  signal_list_m[4],  signal_list_m2[4],
                              signal_list_s[5],  signal_list_m[5],  signal_list_m2[5],
                              signal_list_s[6],  signal_list_m[6],  signal_list_m2[6],
                              signal_list_s[7],  signal_list_m[7],  signal_list_m2[7],
                              signal_list_s[8],  signal_list_m[8],  signal_list_m2[8],
                              signal_list_s[9],  signal_list_m[9],  signal_list_m2[9],
                              signal_list_s[10], signal_list_m[10], signal_list_m2[10],
                              signal_list_s[11], signal_list_m[11], signal_list_m2[11], ), axis=-1)

    signal_stack2 = np.stack((signal_list_l[0],  signal_list_el[0],
                              signal_list_l[1],  signal_list_el[1],
                              signal_list_l[2],  signal_list_el[2],
                              signal_list_l[3],  signal_list_el[3],
                              signal_list_l[4],  signal_list_el[4],
                              signal_list_l[5],  signal_list_el[5],
                              signal_list_l[6],  signal_list_el[6],
                              signal_list_l[7],  signal_list_el[7],
                              signal_list_l[8],  signal_list_el[8],
                              signal_list_l[9],  signal_list_el[9],
                              signal_list_l[10], signal_list_el[10],
                              signal_list_l[11], signal_list_el[11]), axis=-1)

    bigOne = np.concatenate((signal_stack1, signal_stack2), axis=1)

    return bigOne

# This is where I started to see pretty decent results being the third version
# of split and compress, it takes in the most signals and implements the signalCruncher.
def splitAndCompress_Big(rsi_s, rsi_m, rsi_m2, rsi_l, rsi_el,
                         tsi_s, tsi_m, tsi_m2, tsi_l, tsi_el,
                         mfi, bband_l, bband_h):


    rsi_list = signalCruncher(rsi_s, rsi_m, rsi_m2, rsi_l, rsi_el)
    tsi_list = signalCruncher(tsi_s, tsi_m, tsi_m2, tsi_l, tsi_el)

    # Set mfi to be between 0 and 1
    mfi = mfi / 100
    mfi_list = split(mfi)

    # Set stoch to be between 0 and 1
    bband_h_list = split(bband_h)
    bband_l_list = split(bband_l)

    mfi_bband_high_low = np.stack((mfi_list[0],  bband_l_list[0],  bband_h_list[0],
                                   mfi_list[1],  bband_l_list[1],  bband_h_list[1],
                                   mfi_list[2],  bband_l_list[2],  bband_h_list[2],
                                   mfi_list[3],  bband_l_list[3],  bband_h_list[3],
                                   mfi_list[4],  bband_l_list[4],  bband_h_list[4],
                                   mfi_list[5],  bband_l_list[5],  bband_h_list[5],
                                   mfi_list[6],  bband_l_list[6],  bband_h_list[6],
                                   mfi_list[7],  bband_l_list[7],  bband_h_list[7],
                                   mfi_list[8],  bband_l_list[8],  bband_h_list[8],
                                   mfi_list[9],  bband_l_list[9],  bband_h_list[9],
                                   mfi_list[10], bband_l_list[10], bband_h_list[10],
                                   mfi_list[11], bband_l_list[11], bband_h_list[11]), axis=-1)

    sweetJeebus = np.concatenate((rsi_list, tsi_list, mfi_bband_high_low), axis=1)

    return sweetJeebus

# This is the final version (for now) of split and compress and it generate a total of 228 features
# the evolution of the features should be pretty self evident as to how I've added to this function over time.
# I'm leaving the previous functions in this file because different versions of my SVM trader use different
# versions of this function.
def splitAndCompress_Massive(rsi_s, rsi_m, rsi_m2, rsi_l, rsi_el,
                             tsi_s, tsi_m, tsi_m2, tsi_l, tsi_el,
                             mfi_s, mfi_m, mfi_l,
                             bband_l_s, bband_l_m, bband_l_l,
                             bband_h_s, bband_h_m, bband_h_l, ):


    rsi_list = signalCruncher(rsi_s, rsi_m, rsi_m2, rsi_l, rsi_el)
    tsi_list = signalCruncher(tsi_s, tsi_m, tsi_m2, tsi_l, tsi_el)

    # Set mfis to be between 0 and 1
    mfi_s = mfi_s / 100
    mfi_m = mfi_m / 100
    mfi_l = mfi_l / 100

    mfi_list_s = split(mfi_s)
    mfi_list_m = split(mfi_m)
    mfi_list_l = split(mfi_l)

    # Bollinger bands
    bband_h_list_s = split(bband_h_s)
    bband_h_list_m = split(bband_h_m)
    bband_h_list_l = split(bband_h_l)

    bband_l_list_s = split(bband_l_s)
    bband_l_list_m = split(bband_l_m)
    bband_l_list_l = split(bband_l_l)

    mfi_T = np.stack((mfi_list_s[0],  mfi_list_m[0],  mfi_list_l[0],
                      mfi_list_s[1],  mfi_list_m[1],  mfi_list_l[1],
                      mfi_list_s[2],  mfi_list_m[2],  mfi_list_l[2],
                      mfi_list_s[3],  mfi_list_m[3],  mfi_list_l[3],
                      mfi_list_s[4],  mfi_list_m[4],  mfi_list_l[4],
                      mfi_list_s[5],  mfi_list_m[5],  mfi_list_l[5],
                      mfi_list_s[6],  mfi_list_m[6],  mfi_list_l[6],
                      mfi_list_s[7],  mfi_list_m[7],  mfi_list_l[7],
                      mfi_list_s[8],  mfi_list_m[8],  mfi_list_l[8],
                      mfi_list_s[9],  mfi_list_m[9],  mfi_list_l[9],
                      mfi_list_s[10], mfi_list_m[10], mfi_list_l[10],
                      mfi_list_s[11], mfi_list_m[11], mfi_list_l[11]), axis=-1)

    bband_low_T = np.stack((bband_l_list_s[0],  bband_l_list_m[0],  bband_l_list_l[0],
                            bband_l_list_s[1],  bband_l_list_m[1],  bband_l_list_l[1],
                            bband_l_list_s[2],  bband_l_list_m[2],  bband_l_list_l[2],
                            bband_l_list_s[3],  bband_l_list_m[3],  bband_l_list_l[3],
                            bband_l_list_s[4],  bband_l_list_m[4],  bband_l_list_l[4],
                            bband_l_list_s[5],  bband_l_list_m[5],  bband_l_list_l[5],
                            bband_l_list_s[6],  bband_l_list_m[6],  bband_l_list_l[6],
                            bband_l_list_s[7],  bband_l_list_m[7],  bband_l_list_l[7],
                            bband_l_list_s[8],  bband_l_list_m[8],  bband_l_list_l[8],
                            bband_l_list_s[9],  bband_l_list_m[9],  bband_l_list_l[9],
                            bband_l_list_s[10], bband_l_list_m[10], bband_l_list_l[10],
                            bband_l_list_s[11], bband_l_list_m[11], bband_l_list_l[11]), axis=-1)

    bband_high_T = np.stack((bband_h_list_s[0],  bband_h_list_m[0],  bband_h_list_l[0],
                             bband_h_list_s[1],  bband_h_list_m[1],  bband_h_list_l[1],
                             bband_h_list_s[2],  bband_h_list_m[2],  bband_h_list_l[2],
                             bband_h_list_s[3],  bband_h_list_m[3],  bband_h_list_l[3],
                             bband_h_list_s[4],  bband_h_list_m[4],  bband_h_list_l[4],
                             bband_h_list_s[5],  bband_h_list_m[5],  bband_h_list_l[5],
                             bband_h_list_s[6],  bband_h_list_m[6],  bband_h_list_l[6],
                             bband_h_list_s[7],  bband_h_list_m[7],  bband_h_list_l[7],
                             bband_h_list_s[8],  bband_h_list_m[8],  bband_h_list_l[8],
                             bband_h_list_s[9],  bband_h_list_m[9],  bband_h_list_l[9],
                             bband_h_list_s[10], bband_h_list_m[10], bband_h_list_l[10],
                             bband_h_list_s[11], bband_h_list_m[11], bband_h_list_l[11]), axis=-1)

    sweetJeebus = np.concatenate((rsi_list, tsi_list, mfi_T, bband_low_T, bband_high_T), axis=1)

    return sweetJeebus


# Mentioned above, this takes in the X and y and shuffles them together.
def shuffleLists(X, y):
    # Set random state
    rng = RandomState(1234)
    # Shuffle list
    rng.shuffle(X)
    # Set the same random state
    rng = RandomState(1234)
    # Shuffle other list
    rng.shuffle(y)
    return X, y
