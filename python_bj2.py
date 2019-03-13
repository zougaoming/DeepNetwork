#!/usr/bin/env Python
# coding=utf-8
import numpy as np
import zouflow as zf
import torch
from torch import nn,optim
from torch.nn import functional as F


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


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3,
                            out_channels=96,
                            kernel_size=11,
                            stride=4,
                            padding=0)
        self.p1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv1 = nn.Sequential(self.c1,nn.ReLU(),self.p1)


        self.c2 = nn.Conv2d(in_channels=96,
                            out_channels=256,
                            kernel_size=5,
                            padding=2)
        self.p2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Sequential(self.c2,nn.ReLU(),self.p2)


        self.c3 = nn.Conv2d(in_channels=256,
                            out_channels=384,
                            kernel_size=3,
                            padding=1)
        self.conv3 = nn.Sequential(self.c3,nn.ReLU())
        self.c4 = nn.Conv2d(in_channels=384,
                            out_channels=384,
                            kernel_size=3,
                            padding=1)
        self.conv4 = nn.Sequential(self.c4,nn.ReLU())

        self.c5 = nn.Conv2d(in_channels=384,
                            out_channels=256,
                            kernel_size=3,
                            padding=1)
        self.p5 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv5 = nn.Sequential(self.c5,nn.ReLU(),self.p5)

        self.f1 =  nn.Linear(256*6*6, 4096)
        self.fc1 = nn.Sequential(self.f1,nn.ReLU()
                                 #,nn.Dropout(p=0.8)
        )

        self.f2 = nn.Linear(4096, 4096)
        self.fc2 = nn.Sequential(self.f2,nn.ReLU()
                                 #,nn.Dropout(p=0.8)
        )

        self.out = nn.Linear(4096, 10)

    def forward(self, inputs):
        network = self.conv1(inputs)
        self.conv1_o = network
        network = self.conv2(network)
        self.conv2_o = network
        network = self.conv3(network)
        self.conv3_o = network
        network = self.conv4(network)
        self.conv4_o = network
        network = self.conv5(network)
        self.conv5_o = network
        network = network.view(-1, 256 * 6 * 6)
        #network = network.view(network.size(0), -1)
        network = self.fc1(network)
        network = self.fc2(network)
        self.fc2_o = network
        out = self.out(network)
        self.lastout = out
        return out
if __name__ == '__main__':

    BATCH_SIZE = 1
    x_train = np.random.uniform(-0.1,0.1,size=(BATCH_SIZE,3,227,227))
    x_train = x_train.astype(np.float32)
    y_train = np.random.uniform(-0.1,0.1,size=(BATCH_SIZE,10))
    y_train = y_train.astype(np.float32)
    param = zf.Param('./gojs/AlexNet.txt')
    network = zf.Network(param=param)

    model = AlexNet()
    criterion = nn.MSELoss()
    param.Loss = zf.MSELoss()
    param.rate = 1.0
    param.Optimizer = zf.AdadeltaOptimizer(param)
    #optimizer = optim.Adam(model.parameters(), lr=0.05,betas=(0.9,0.99),eps=1e-10)
    #optimizer = optim.SGD(model.parameters(),lr=0.05)
    optimizer = optim.Adadelta(model.parameters(),lr=1.0,rho=0.95,eps=1e-10)
    for e in range(100):
        for offset in range(1):
            print(e,'-------------',offset)

            x_tensor = torch.tensor(x_train, requires_grad=True)
            y_tensor = torch.tensor(y_train, requires_grad=False)
            out = model(x_tensor)


            #print('conv1andpool->', model.conv1andpool)
            #print('conv2andpool->', model.conv2andpool)
            #print('flatern->', model.flatern)


            conv1_w = model.c1.weight.data.numpy()
            conv1_b = model.c1.bias.data.numpy()

            conv2_w = model.c2.weight.data.numpy()
            conv2_b = model.c2.bias.data.numpy()
            conv3_w = model.c3.weight.data.numpy()
            conv3_b = model.c3.bias.data.numpy()
            conv4_w = model.c4.weight.data.numpy()
            conv4_b = model.c4.bias.data.numpy()
            conv5_w = model.c5.weight.data.numpy()
            conv5_b = model.c5.bias.data.numpy()

            fc1_w = model.f1.weight.data.numpy()
            fc1_b = model.f1.bias.data.numpy()

            fc2_w = model.f2.weight.data.numpy()
            fc2_b = model.f2.bias.data.numpy()

            fc3_w = model.out.weight.data.numpy()
            fc3_b = model.out.bias.data.numpy()

            if offset == 0  and e == 0:
                network.GetSet.setW('CNN1',conv1_w)
                network.GetSet.setBias('CNN1', conv1_b)

                network.GetSet.setW('CNN2',conv2_w)
                network.GetSet.setBias('CNN2',conv2_b)

                network.GetSet.setW('CNN3', conv3_w)
                network.GetSet.setBias('CNN3', conv3_b)
                network.GetSet.setW('CNN4', conv4_w)
                network.GetSet.setBias('CNN4', conv4_b)
                network.GetSet.setW('CNN5', conv5_w)
                network.GetSet.setBias('CNN5', conv5_b)

                network.GetSet.setW('Neuron1',fc1_w.T)
                network.GetSet.setBias('Neuron1', fc1_b.reshape(-1,1))
                network.GetSet.setW('Neuron2', fc2_w.T)
                network.GetSet.setBias('Neuron2', fc2_b.reshape(-1, 1))
                network.GetSet.setW('Neuron3', fc3_w.T)
                network.GetSet.setBias('Neuron3', fc3_b.reshape(-1, 1))

            network.forward(x_train)
            network.setOutputDelta(y_train.T)
            network.backward()


            print(max(network.GetSet.getOutput('Neuron2').T - model.fc2_o.data.numpy()))
            loss = criterion(out, y_tensor)
            print('loss->', loss.data.numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()










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