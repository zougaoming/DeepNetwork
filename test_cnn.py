#!/usr/bin/env Python
# coding=utf-8
import numpy as np
from NetworkByJson import Param,Network

if __name__ == '__main__':

	X = np.array([[.1, .1, .9, .4, .5, .6, .1, .5],
				  [.1, .2, .8, .7, .0, .6, .7, .1],
				  [.4, .7, .6, .4, .5, .6, .3, .5],
				  [.1, .2, .3, .8, .7, .8, .1, .7],
				  [.1, .9, .8, .4, .5, .6, .6, .5],
				  [.5, .2, .3, .9, .5, .0, .1, .0],
				  [.1, .1, .9, .4, .5, .3, .8, .5],
				  [.1, .2, .2, .4, .8, .6, .1, .5]]
				 , dtype=np.float64)
	Y = np.array([[.4, .5, .0,.1,.6,.0],
				  [.3, .5, .0,.1,.2,.1],
				  [.4, .4, .7,.1,.2,.3],
				  [.4,.5,.0,.2,.2,.9],
				  [.4,.5,.0,.1,.2,.5],
				  [.4,.5,.0,.1,.2,.6]])

	Y = np.array([[.4, .5,   .6, .0,.2],
				  [.3, .5,   .2, .1,.3],
				  [.4, .4,   .2, .3,.5],
				  [.4, .5,   .2, .9,.7],
				  [.1, .3, .4, .8, .0]])
	w_init = np.array([[[.1, .2, .4],
						[.3, .6, .7],
						[.5, .2, .8]]], dtype=np.float64)
	Y2 = np.array([[.4, .5, .2],
				  [0, .1, .2],
				  [.1, .3, .2]])

	w_init2 = np.array([[[.1, .2],
						 [.3, .6]
						 ]], dtype=np.float64)
	Y1 = [[[0.6]]]
	param = Param('./gojs/cnn2.txt')
	network = Network(param=param)

	X = X.reshape((1,1,X.shape[0],X.shape[1]))
	network.param.w_1 = w_init.reshape((1,1,3,3))
	network.param.w_12 = w_init2.reshape((1,1,2,2))

	#network.param.w_2 = _w1_init
	#network.param.w_3 = _w2_init
	#network.param.w_6 = _w3_init
	#X = X.reshape((1,X.shape[0],X.shape[1]))
	for e in range(2):
		print('-----------',e)
		network.forward(X)
		t = network.Output
		print(t)
		network.setOutputDelta(Y1)
		network.backward()
