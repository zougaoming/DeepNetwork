#!/usr/bin/env Python
# coding=utf-8
import numpy as np
from NetworkByJson import Param,Network

if __name__ == '__main__':

	x = np.array([[.1, .2, .3, .4, .5, .6]])
	y = np.array([[.4, .5, 0, .1]])
	_w1_init = np.array([[.1, .2, .3, .4],
						 [.1, .2, .3, .4],
						 [.1, .2, .3, .4],
						 [.1, .2, .3, .4],
						 [.1, .2, .3, .4],
						 [.1, .2, .3, .4]], dtype=np.float32)
	_w2_init = np.array([[.2, .2, .3, .1],
						 [.2, .2, .3, .1],
						 [.2, .2, .3, .1],
						 [.2, .2, .3, .1]], dtype=np.float32)

	_w3_init = np.array([[.1, .8, .3, .1],
						 [.2, .4, .2, .3],
						 [.3, .6, .5, .1],
						 [.4, .2, .3, .7]], dtype=np.float32)

	param = Param('./gojs/model3.txt')
	network = Network(param=param)

	network.param.w_2 = _w1_init
	network.param.w_3 = _w2_init
	network.param.w_6 = _w3_init
	for e in range(100):
		print('-----------')
		network.forward(x.T)
		t = network.Output
		print(t)
		network.setOutputDelta(y.T)
		network.backward()
