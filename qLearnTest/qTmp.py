import numpy as np
from csvGen import CSVStreamer

c = CSVStreamer._generator('../largeData/data.data')

for i in range(10):
	print(next(c))