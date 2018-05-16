import requests
import sys
import time
from datetime import datetime, timedelta

def getAllData(coin):
	baseURL = 'https://api.gdax.com/'
	tmp = 'products/'
	tmp2 = 'candles/'
	date = datetime(2016,1,1)
	date2 = datetime(2018,5,1)
	f = open('priceData/' + date.isoformat('T').split('-')[0] + '/' + coin + '1' + 'min.data','w')
	cm = 1
	URL = baseURL + tmp + coin + '/' + tmp2
	while date<date2:
		start = time.time()
		lastDate = date
		date += timedelta(hours=4)
		print(lastDate.isoformat('T').split('-')[1])
		if cm != int(lastDate.isoformat('T').split('-')[1]):
			f.close()
			cm+=1
			f = open('priceData/' + lastDate.isoformat('T').split('-')[0] + '/' + coin + str(cm) + 'min.data', 'w')
			cm-=1
			cm=cm%12
			cm+=1
		print(lastDate.isoformat('T') + '.000Z')
		payload = {'start':lastDate, 'end':date}
		t = requests.get(URL, params=payload).text
		t2 = t.split('[')
		t2=t2[::-1]
		for i in t2:
			i = i.replace(']','')
			if i != '':
		 		f.write(i[:-1] + '\n')
		while time.time()-start < 0.8:
			time.sleep(0.1)
		#print(date)

pairs = ['BTC-USD','BCH-USD','ETH-USD','LTC-USD']
for i in pairs:
	getAllData(i)

#t = requests.get(URL, params=payload).text
#t2 = t.split('[')
#for i in t2:
#	i = i.replace(']','').split(',')
#	for j in i:
#		if j != '':
#			sys.stdout.write(j + ', ')
#	print()

