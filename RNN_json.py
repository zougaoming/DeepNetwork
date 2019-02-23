#!/usr/bin/env Python
# coding=utf-8
import numpy as np
from NetworkByJson import Param,Network

def data_set():
	x = [np.array([[1], [0], [1]]),
         np.array([[2], [3], [4]])]
	d = np.array([[.12], [.88]])
	return x,d

def tt(cache):
	c,d = cache
	print(c,d)
if __name__ == '__main__':

	param = Param('./gojs/RNN.txt')
	network = Network(param=param)
	x, y = data_set()
	for e in range(1000):
		print('-----------')
		network.forward(x[0])
		t = network.Output
		print(t)
		network.setOutputDelta(y)
		network.backward()
