import pandas as pd
import talib as EMA
import matplotlib as plot





raw_2016_Jan = pd.read_csv("priceData/2016/BTC-USD1min.data", header=None)
raw_2016_Feb = pd.read_csv("priceData/2016/BTC-USD2min.data", header=None)
raw_2016_Mar = pd.read_csv("priceData/2016/BTC-USD3min.data", header=None)
raw_2016_Apr = pd.read_csv("priceData/2016/BTC-USD4min.data", header=None)
raw_2016_May = pd.read_csv("priceData/2016/BTC-USD5min.data", header=None)
raw_2016_Jun = pd.read_csv("priceData/2016/BTC-USD6min.data", header=None)
raw_2016_Jul = pd.read_csv("priceData/2016/BTC-USD7min.data", header=None)
raw_2016_Aug = pd.read_csv("priceData/2016/BTC-USD8min.data", header=None)
raw_2016_Sep = pd.read_csv("priceData/2016/BTC-USD9min.data", header=None)
raw_2016_Oct = pd.read_csv("priceData/2016/BTC-USD10min.data", header=None)
raw_2016_Nov = pd.read_csv("priceData/2016/BTC-USD11min.data", header=None)
raw_2016_Dec = pd.read_csv("priceData/2016/BTC-USD12min.data", header=None)

raw_2016_Dec.columns = ['Time','Low','High','Open','Close','Volume']

#raw_2016_Jan_close = raw_2016_Jan['4']

#output = EMA(raw_2016_Jan[,4], timepierod= 30)



#print(raw_2016_Jan_close.tail())

print(raw_2016_Dec.tail())



#rawData.rename(columns={'0':'Time', '1':'Open','2': 'High','3': 'Low','4': 'Close','5': 'Volume','6': 'CloseTime',
#                        '7': 'Quote_Asset_Volume','8': 'Num_of_Trades','9': 'TBQAV','10': 'TBQAV','11': 'Ignore',})

#rawData.columns = ['Time','Open','High','Low','Close','Volume','CloseTime', 'Quote_Asset_Volume',
 #                  'Num_of_Trades','TBQAV','TBQAV', 'Ignore']



#print(rawData.head())