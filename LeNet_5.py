#!/usr/bin/env Python
# coding=utf-8
import numpy as np
from NetworkByJson import Param,Network
from tensorflow.examples.tutorials.mnist import input_data
#from data.mnist.mnist import MNIST_DATA
from sklearn.utils import shuffle
import tensorflow as tf
def printImage(image):
    for i in range(28):
        for j in range(28):
            if image[i][j] == 0 :
                print(' ',end='')
            else:
                print('*',end='')
        print('')

if __name__ == '__main__':

	#mnist = input_data.read_data_sets("data/mnist/", reshape=False)
	#x_train, Y = mnist.train.images[0], mnist.train.labels[0]
	#X = np.pad(x_train, ((2, 2), (2, 2),(0,0)), 'constant')
	#data = MNIST_DATA('./data/mnist/')
	#X, Y, _ = data.get_train_data(0, 5000)
	#X = X.astype(np.float32)
	#Y = Y.T
	#b = 0
	#data.printImage(X[0, :, :])

	#X = np.pad(X, ((0,0),(2, 2), (2, 2)), 'constant')
	#X = X.reshape(5000,32,32)
	#Y = Y.reshape(5000, 10, 1)
	#x_validation = np.pad(x_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
	mnist = input_data.read_data_sets("data/mnist/", reshape=False)
	x_train, y_train = mnist.train.images, mnist.train.labels
	x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
	param = Param('./gojs/LeNet-5.txt')
	network = Network(param=param)

	#w1_init = w1_init.reshape(1,6,5,5)
	#network.param.w_1 = w1_init
	BATCH_SIZE = 128
	#network.param.w_2 = _w1_init
	#network.param.w_3 = _w2_init
	#network.param.w_6 = _w3_init
	for e in range(10):
		x_train, y_train = shuffle(x_train, y_train)
		for offset in range(0, len(x_train), BATCH_SIZE):
			end = offset + BATCH_SIZE
			batch_x, batch_y = x_train[offset:end], y_train[offset:end]
			#y_train = tf.one_hot(y_train, 10)
			batch_x = batch_x.reshape(BATCH_SIZE,1,32,32)
			#one_hot_y = np.zeros((10, BATCH_SIZE))
			#print('-----------',e)
			network.forward(batch_x)
			t = network.Output
			print('Y->',batch_y,'out->',t)
			#print(t)
			#for i in range(BATCH_SIZE):
			one_hot_y = tf.one_hot(batch_y, 10)
			one_hot_y.reshape(10,BATCH_SIZE)
			network.setOutputDelta(one_hot_y)
			network.backward()
