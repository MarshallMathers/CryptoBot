import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ta
import time
from logbook import Logger

from keras.models import model_from_json
from keras.utils import to_categorical

from catalyst import run_algorithm
from catalyst.api import (record, symbol, order_target_percent,)
from catalyst.exchange.utils.stats_utils import extract_transactions

NAMESPACE = 'bollinger_bands'
log = Logger(NAMESPACE)

def normalize(inData):
	std = np.std(inData)
	mean = np.mean(inData)
	out=[]
	for i in range(len(inData)):
		t = ((inData[i]-mean)/std)
		out.append(t)
	return np.array(out)

def initialize(context):
	context.i = 0
	context.asset = symbol('btc_usd')
	context.base_price = None
	
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	context.model = model_from_json(loaded_model_json)
	# load weights into new model
	context.model.load_weights("model.h5")
	context.lastTrade=0
	print("Loaded model from disk")
	
	
def handle_data(context, data):
#	start = time.time()
	#NN was trained with 15 min CandleSticks
	csTime=15
	startTime = 144
	startMin = csTime*startTime
	
	#signals to collect for NN	
	#keltner uppand and lower
	keltN    = 15
	#RSI
	rsiN      = 14
	#EMA fast and EMA slow
	emaFN = 12
	emaSN = 26
	#macd slow and fast N
	macdFN = 12
	macdSN = 26
	#vortex Indicator
	ichin1 = 9
	ichin2 = 26
	ichin3 = 52

	
	# Skip as many bars as startMin to properly compute the average
	context.i += 1
	timeStamp = 360
	#if context.i % timeStamp != 0:
	#	return
	if context.i % timeStamp == 0:
		print(context.i/1440)
	if context.i < startMin:
		return
		
	if context.i % 5 != 0:
		return
	# Since we are using limit orders, some orders may not execute immediately
	# we wait until all orders are executed before considering more trades.
	orders = context.blotter.open_orders
	if len(orders) > 0:
		return

	# Exit if we cannot trade
	if not data.can_trade(context.asset):
		return
	mN = max([keltN,rsiN,emaFN,emaSN,macdFN,macdSN,ichin1,ichin2,ichin3])
	close = data.history(context.asset,'close',bar_count=startMin,frequency='15T')
	low = data.history(context.asset,'low',bar_count=startMin,frequency='15T')
	high = data.history(context.asset,'high',bar_count=startMin,frequency='15T')
	volume = data.history(context.asset, 'volume', bar_count=startMin, frequency='15T')
	price = data.current(context.asset, 'price')


	if context.i % timeStamp == 0:
		print(price)
	inData=[[],[],[],[],[],[],[],[],[]]


	inData[0] = np.array(list(ta.momentum.rsi(close, n=rsiN))) / 100

	inData[1] = np.array(list(ta.momentum.tsi(close, r=25, s=13))) / 100

	inData[2] = np.array(list(ta.momentum.rsi(close, n=rsiN+5))) / 100

	inData[3] = np.array(list(ta.momentum.tsi(close, r=28, s=15))) / 100

	inData[4] = np.array(list(ta.trend.ema_fast(close, n_fast=emaFN)))

	inData[5] = np.array(list(ta.trend.ema_slow(close, n_slow=emaSN)))

	inData[6] = np.array(list(ta.volatility.keltner_channel_hband(high, low, close, n=keltN)))

	inData[7] = np.array(list(ta.volatility.keltner_channel_lband(high, low, close, n=keltN)))

	inData[8] = np.array(close)


	cl = inData[8]
	for i in range(len(inData)):
		inData[i] = inData[i][mN:]

	for i in range(len(inData[4])):
		inData[4][i] = (cl[i] - inData[4][i]) / inData[4][i] * 100

	for i in range(len(inData[5])):
		inData[5][i] = (cl[i] - inData[5][i]) / inData[5][i] * 100

	for i in range(len(inData[6])):
		inData[6][i] = (cl[i] - inData[6][i]) / inData[6][i] * 100

	for i in range(len(inData[7])):
		inData[7][i] = (cl[i] - inData[7][i]) / inData[7][i] * 100

	x = []
	
	lstm1Len=24
	for i in range(len(inData[0])-lstm1Len-1,len(inData[0])-1):
		currX = []
		for j in range(len(inData)-1):
			currX.append(inData[j][i])
		x.append(currX)

	npx = np.array(x)
	npx = npx.reshape((1,lstm1Len,8))
	y = context.model.predict_on_batch(npx)
	
	y = y[0]
#	if context.i % 10 ==0:
#		print(y)
	t1=0
	t2=0
	t3=0
#	if y[0] > y[1]:
#		t1=1
#		t2=0
#	if y[0] < y[1]:
#		t1=0
#		t2=1
	
		

#	sellVal = -0.5
#	buyVal = 0.5
#	if y>buyVal:
#		t1=1
#		t2=0
#	elif y<sellVal:
#		t1=0
#		t2=1
#	else:
#		t1=0
#		t2=0
#		
#	t1 = 1 if y[0] > 0.5 else 0
#	t2 = 1 if y[1] > 0.5 else 0

	#t1 = 1 if y[0] > 0.5 else 0
	#t3 = 1 if y[1] > 0.5 else 0
	#t2 = 1 if y[2] > 0.5 else 0


	t1 = 1 if y >= 0.5 else 0
	t3 = 1 if y < 0.5 else 0


#	t1 = 1 if y[1] > 0.5 else 0
#	t2 = 1 if y[0] > 0.5 else 0
	
	
	#t1=0
	#t2=1
	#t3 = 1 if y[0][2] > 0.5 else 0
	#print(t1,t2)#,t3)
	#print()
	#print(t1,t2,t3)
	#print()
	# If base_price is not set, we use the current value. This is the
	# price at the first bar which we reference to calculate price_change.
	if context.base_price is None:
		context.base_price = price
	price_change = (price - context.base_price) / context.base_price

	# Save values for later inspection
	record(price=price,
		   cash=context.portfolio.cash,
		   price_change=price_change)
	if t1==t3:
		return


	# We check what's our position on our portfolio and trade accordingly
	pos_amount = context.portfolio.positions[context.asset].amount
	tradeTime = 60
	if context.i < context.lastTrade+tradeTime:
		return
	if context.i % 5 != 0:
		return
		
	if t1==1 and pos_amount == 0:
		log.info('{}: buying - price: {}'.format(data.current_dt, price))
		
		# Set a style for limit orders,
		limit_price = price * 1.005
		order_target_percent(
			context.asset, 1, limit_price=limit_price
		)

	elif t3==1 and pos_amount > 0:
		log.info('{}: selling - price: {}'.format(data.current_dt, price))
		limit_price = price * 0.995
		order_target_percent(
		    context.asset, 0, limit_price=limit_price
		)
	# Trading logic
#	if t1==1 and pos_amount == 0:
#		context.lastTrade = context.i
#		print('buy : ',price)
#		order_target_percent(context.asset, 1)
#	elif t2==1 and pos_amount > 0:
#		context.lastTrade = context.i
#		order_target_percent(context.asset, 0)
#		print('sell: ',price)
		
	
	#if short_mavg > long_mavg and pos_amount == 0:
		# we buy 100% of our portfolio for this asset
	#	order_target_percent(context.asset, 1)
	#elif short_mavg < long_mavg and pos_amount > 0:
		# we sell all our positions for this asset
	#	order_target_percent(context.asset, 0)
#	print((time.time()-start)*1000,'milliseconds')
def analyze(context, perf):
	# Get the base_currency that was passed as a parameter to the simulation
	exchange = list(context.exchanges.values())[0]
	base_currency = exchange.base_currency.upper()

	# First chart: Plot portfolio value using base_currency
	ax1 = plt.subplot(411)
	perf.loc[:, ['portfolio_value']].plot(ax=ax1)
	ax1.legend_.remove()
	ax1.set_ylabel('Portfolio Value\n({})'.format(base_currency))
	start, end = ax1.get_ylim()
	ax1.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

	# Second chart: Plot asset price, moving averages and buys/sells
	ax2 = plt.subplot(412, sharex=ax1)
	perf.loc[:, ['price', 'short_mavg', 'long_mavg']].plot(
		ax=ax2,
		label='Price')
	ax2.legend_.remove()
	ax2.set_ylabel('{asset}\n({base})'.format(
		asset=context.asset.symbol,
		base=base_currency
	))
	start, end = ax2.get_ylim()
	ax2.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

	transaction_df = extract_transactions(perf)
	if not transaction_df.empty:
		buy_df = transaction_df[transaction_df['amount'] > 0]
		sell_df = transaction_df[transaction_df['amount'] < 0]
		ax2.scatter(
			buy_df.index.to_pydatetime(),
			perf.loc[buy_df.index, 'price'],
			marker='^',
			s=100,
			c='green',
			label=''
		)
		ax2.scatter(
			sell_df.index.to_pydatetime(),
			perf.loc[sell_df.index, 'price'],
			marker='v',
			s=100,
			c='red',
			label=''
		)

	# Third chart: Compare percentage change between our portfolio
	# and the price of the asset
	ax3 = plt.subplot(413, sharex=ax1)
	perf.loc[:, ['algorithm_period_return', 'price_change']].plot(ax=ax3)
	ax3.legend_.remove()
	ax3.set_ylabel('Percent Change')
	start, end = ax3.get_ylim()
	ax3.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

	# Fourth chart: Plot our cash
	ax4 = plt.subplot(414, sharex=ax1)
	perf.cash.plot(ax=ax4)
	ax4.set_ylabel('Cash\n({})'.format(base_currency))
	start, end = ax4.get_ylim()
	ax4.yaxis.set_ticks(np.arange(0, end, end / 5))

	plt.show()


if __name__ == '__main__':
	
	run_algorithm(
			capital_base=10000,
			data_frequency='minute',
			initialize=initialize,
			handle_data=handle_data,
			analyze=analyze,
			exchange_name='bitfinex',
			algo_namespace=NAMESPACE,
			base_currency='usd',
			start=pd.to_datetime('2018-1-1', utc=True),
			end=pd.to_datetime('2018-3-28', utc=True),
		)
