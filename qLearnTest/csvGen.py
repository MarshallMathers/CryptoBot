import csv

import numpy as np


class csvStream():
		
	@staticmethod
	def _generator(fn):
		with open(fn, "r") as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				out = []
				out.append(float(row[3]))
				out.append(float(row[1]))
				out.append(float(row[2]))
				out.append(float(row[5]))
				out.append(float(row[7]))
				out.append(float(row[10]))
				#print(len(row))
				yield np.array(row, dtype=np.float)	