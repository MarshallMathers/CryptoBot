import re



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


