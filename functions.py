import re
import numpy as np
from numpy.random import RandomState


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


def splitAndCompress(close, rsi, tsi, mfi, macdSig):
    train_Total = close

    close_00 = close[::12]
    close_01 = close[1::12]
    close_02 = close[2::12]
    close_03 = close[3::12]
    close_04 = close[4::12]
    close_05 = close[5::12]
    close_06 = close[6::12]
    close_07 = close[7::12]
    close_08 = close[8::12]
    close_09 = close[9::12]
    close_10 = close[10::12]
    close_11 = close[11::12]

    # Set rsi to be between 0 and 1
    rsi = rsi/100
    rsi_00 = rsi[::12]
    rsi_01 = rsi[1::12]
    rsi_02 = rsi[2::12]
    rsi_03 = rsi[3::12]
    rsi_04 = rsi[4::12]
    rsi_05 = rsi[5::12]
    rsi_06 = rsi[6::12]
    rsi_07 = rsi[7::12]
    rsi_08 = rsi[8::12]
    rsi_09 = rsi[9::12]
    rsi_10 = rsi[10::12]
    rsi_11 = rsi[11::12]

    # Set tsi to be between 0 and 1
    tsi = tsi/100
    tsi_00 = tsi[::12]
    tsi_01 = tsi[1::12]
    tsi_02 = tsi[2::12]
    tsi_03 = tsi[3::12]
    tsi_04 = tsi[4::12]
    tsi_05 = tsi[5::12]
    tsi_06 = tsi[6::12]
    tsi_07 = tsi[7::12]
    tsi_08 = tsi[8::12]
    tsi_09 = tsi[9::12]
    tsi_10 = tsi[10::12]
    tsi_11 = tsi[11::12]

    # Set mfi to be between 0 and 1
    mfi = mfi / 100
    mfi_00 = mfi[::12]
    mfi_01 = mfi[1::12]
    mfi_02 = mfi[2::12]
    mfi_03 = mfi[3::12]
    mfi_04 = mfi[4::12]
    mfi_05 = mfi[5::12]
    mfi_06 = mfi[6::12]
    mfi_07 = mfi[7::12]
    mfi_08 = mfi[8::12]
    mfi_09 = mfi[9::12]
    mfi_10 = mfi[10::12]
    mfi_11 = mfi[11::12]


    macdSig_00 = macdSig[::12]
    macdSig_01 = macdSig[1::12]
    macdSig_02 = macdSig[2::12]
    macdSig_03 = macdSig[3::12]
    macdSig_04 = macdSig[4::12]
    macdSig_05 = macdSig[5::12]
    macdSig_06 = macdSig[6::12]
    macdSig_07 = macdSig[7::12]
    macdSig_08 = macdSig[8::12]
    macdSig_09 = macdSig[9::12]
    macdSig_10 = macdSig[10::12]
    macdSig_11 = macdSig[11::12]

    train_Total = np.stack((close_00, rsi_00, tsi_00, mfi_00, macdSig_00,
                            close_01, rsi_01, tsi_01, mfi_01, macdSig_01,
                            close_02, rsi_02, tsi_02, mfi_02, macdSig_02,
                            close_03, rsi_03, tsi_03, mfi_03, macdSig_03,
                            close_04, rsi_04, tsi_04, mfi_04, macdSig_04,
                            close_05, rsi_05, tsi_05, mfi_05, macdSig_05,
                            close_06, rsi_06, tsi_06, mfi_06, macdSig_06,
                            close_07, rsi_07, tsi_07, mfi_07, macdSig_07,
                            close_08, rsi_08, tsi_08, mfi_08, macdSig_08,
                            close_09, rsi_09, tsi_09, mfi_09, macdSig_09,
                            close_10, rsi_10, tsi_10, mfi_10, macdSig_10,
                            close_11, rsi_11, tsi_11, mfi_11, macdSig_11), axis=-1)

    return train_Total


def split(ta_signal):
    out = []
    for i in range(12):
        out.append(ta_signal[i::12])
    return out


def splitAndCompress_noPrice(rsi_s, tsi_s, rsi_l, tsi_l, mfi, stochSig):

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
    stochSig = stochSig / 100
    stochSig_list = split(stochSig)

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

    mfi_stoch_stochSig = np.stack((mfi_list[0],  stochSig_list[0],
                                   mfi_list[1],  stochSig_list[1],
                                   mfi_list[2],  stochSig_list[2],
                                   mfi_list[3],  stochSig_list[3],
                                   mfi_list[4],  stochSig_list[4],
                                   mfi_list[5],  stochSig_list[5],
                                   mfi_list[6],  stochSig_list[6],
                                   mfi_list[7],  stochSig_list[7],
                                   mfi_list[8],  stochSig_list[8],
                                   mfi_list[9],  stochSig_list[9],
                                   mfi_list[10], stochSig_list[10],
                                   mfi_list[11], stochSig_list[11]), axis=-1)

    sweetJeebus = np.concatenate((rsi_tsi_s_l, mfi_stoch_stochSig), axis=1)

    return sweetJeebus


