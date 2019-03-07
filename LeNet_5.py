#!/usr/bin/env Python
# coding=utf-8
import numpy as np
from NetworkByJson import Param,Network
from tensorflow.examples.tutorials.mnist import input_data
#from data.mnist.mnist import MNIST_DATA
from sklearn.utils import shuffle
import torch
from torch import nn,optim
from torch.nn import functional as F
from torch.autograd import Variable
import tensorflow as tf
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

#定义lenet5
class LeNet5(nn.Module):
    def __init__(self):
        '''构造函数，定义网络的结构'''
        super().__init__()
        #定义卷积层，1个输入通道，6个输出通道，5*5的卷积filter，外层补上了两圈0,因为输入的是32*32
        self.conv1 = nn.Conv2d(1, 6, 5, padding=0)
        #第二个卷积层，6个输入，16个输出，5*5的卷积filter
        self.conv2 = nn.Conv2d(6, 16, 5)

        #最后是三个全连接层
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        '''前向传播函数'''
        #先卷积，然后调用relu激活函数，再最大值池化操作
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        self.conv1andpool = x
        #第二次卷积+池化操作
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        self.conv2andpool = x
        #重新塑形,将多维数据重新塑造为二维数据，256*400
        x = x.view(-1, self.num_flat_features(x))
        self.flatern = x
        #print('size', x.size())
        #第一个全连接
        x = F.relu(self.fc1(x))
        self.fc1o = x
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        #x.size()返回值为(256, 16, 5, 5)，size的值为(16, 5, 5)，256是batch_size
        size = x.size()[1:]        #x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
if __name__ == '__main__':

    mnist = input_data.read_data_sets("data/mnist/", reshape=False)
    x_train, y_train = mnist.train.images, mnist.train.labels
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    param = Param('./gojs/LeNet-5.txt')
    network = Network(param=param)
    # w1_init = w1_init.reshape(1,6,5,5)
    # network.param.w_1 = w1_init
    BATCH_SIZE = 128
    #param.rate = param.rate / BATCH_SIZE




    model = LeNet5()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    for e in range(2):
        x_train, y_train = shuffle(x_train, y_train)
        #for offset in range(1):
        for offset in range(0, len(x_train), BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            # y_train = tf.one_hot(y_train, 10)
            batch_x = batch_x.reshape(BATCH_SIZE, 1, 32, 32)
            one_hot_y = dense_to_one_hot(batch_y, 10)



            '''
            x_tensor = torch.tensor(batch_x, requires_grad=True)
            y_tensor = torch.tensor(one_hot_y, requires_grad=True)
            out = model(x_tensor)

            loss = criterion(out, y_tensor)
            print('out->', out.data.numpy(),y_tensor)
            print('loss->', loss)
            #print('conv1andpool->', model.conv1andpool)
            #print('conv2andpool->', model.conv2andpool)
            #print('flatern->', model.flatern)


            conv1_w = model.conv1.weight.data.numpy()
            conv1_b = model.conv1.bias.data.numpy()

            conv2_w = model.conv2.weight.data.numpy()
            conv2_b = model.conv2.bias.data.numpy()

            fc1_w = model.fc1.weight.data.numpy()
            fc1_b = model.fc1.bias.data.numpy()

            fc2_w = model.fc2.weight.data.numpy()
            fc2_b = model.fc2.bias.data.numpy()

            fc3_w = model.fc3.weight.data.numpy()
            fc3_b = model.fc3.bias.data.numpy()

            if offset == 0:
                network.param.w_1 = conv1_w
                network.param.b_1 = conv1_b
                network.param.w_11 = conv2_w
                network.param.b_11 = conv2_b

                network.param.w_18 = fc1_w.T
                network.param.b_18 = fc1_b.reshape(-1,1)
                network.param.w_19 = fc2_w.T
                network.param.b_19 = fc2_b.reshape(-1,1)
                network.param.w_20 = fc3_w.T
                network.param.b_20 = fc3_b.reshape(-1,1)

            '''
            network.forward(batch_x)

            print('zj->',network.Output[:,0],one_hot_y[0])
            one_hot_y = one_hot_y.reshape(10,BATCH_SIZE)
            network.setOutputDelta(one_hot_y)
            network.backward()
            #print('conv1andpool->',network.o_8)

            #print('conv2andpool->',network.o_14)
            #print('flatern->', network.o_17)

            '''

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''









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