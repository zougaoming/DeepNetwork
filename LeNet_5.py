#!/usr/bin/env Python
# coding=utf-8
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import zouflow as zf
from data.mnist.mnist import MNIST_DATA
from sklearn.utils import shuffle
import dill

def printImage(image):
    for i in range(28):
        for j in range(28):
            if image[i][j] == 0 :
                print(' ',end='')
            else:
                print('*',end='')
        print('')


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes),dtype=np.float32)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

if __name__ == '__main__':

    mnist = input_data.read_data_sets("data/mnist/", reshape=False)
    x_train, y_train = mnist.train.images, mnist.train.labels
    #x_train,y_train,_ = MNIST_DATA("data/mnist/").get_train_data(0,60000)
    #x_train = x_train.astype(np.float64)
    #y_train.astype(np.float64)
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    #x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant')

    #gfile.Exists(work_directory):
    param = zf.Param('./gojs/LeNet-5.txt')
    network = zf.Network(param=param)
    save = zf.Save('./save_d.pkl', param)
    save.load()
    #dill.load_session('./save_d.pkl', main=network)
    # w1_init = w1_init.reshape(1,6,5,5)
    # network.param.w_1 = w1_init
    BATCH_SIZE = 128
    #param.rate = param.rate / BATCH_SIZE




    param.Loss = zf.MSELoss()
    param.rate = 0.1
    param.Optimizer = zf.SGDandMomentumOptimizer(param)
    for e in range(100):
        print('--------------',e)
        x_train, y_train = shuffle(x_train, y_train)
        #for offset in range(1):
        for offset in range(0, len(x_train), BATCH_SIZE):
            print(e,'-------------',offset)
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            # y_train = tf.one_hot(y_train, 10)
            batch_x = batch_x.reshape(-1, 1, 32, 32)
            batch_y = dense_to_one_hot(batch_y, 10)


            network.forward(batch_x)


            print('out->',network.Output[:,0:2],batch_y[0:2])

            network.setOutputDelta(batch_y.T)
            network.backward()
            save.store()













    '''
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
            print('Y->',batch_y,'out->',t[:,0])
            #print(t)
            #for i in range(BATCH_SIZE):
            one_hot_y = dense_to_one_hot(batch_y, 10)
            one_hot_y = one_hot_y.reshape(10,BATCH_SIZE)
            network.setOutputDelta(one_hot_y)
            network.backward()
    '''