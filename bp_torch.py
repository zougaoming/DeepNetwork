# !/usr/bin/python
# coding: utf8
from torch import nn,optim
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

class simpleNet(nn.Module):
    """
    定义了一个简单的三层全连接神经网络，每一层都是线性的
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Activation_Net(nn.Module):
    """
    在上面的simpleNet的基础上，在每层的输出部分添加了激活函数
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.l1 = nn.Linear(in_dim, n_hidden_1,bias=True)
        self.layer1 = nn.Sequential(self.l1, nn.Tanh())
        self.l2 = nn.Linear(n_hidden_1, n_hidden_2,bias=True)
        self.layer2 = nn.Sequential(self.l2, nn.Tanh())
        self.l3 = nn.Linear(n_hidden_2, out_dim,bias=True)
        self.layer3 = nn.Sequential(self.l3,nn.Tanh())
        """
        这里的Sequential()函数的功能是将网络的层组合到一起。
        """
        _w1_init = np.array([[.1, .2, .3, .4],
                             [.1, .2, .3, .4],
                             [.1, .2, .3, .4],
                             [.1, .2, .3, .4],
                             [.1, .2, .3, .4],
                             [.1, .2, .3, .4]],dtype=np.float32)
        _w2_init = np.array([[.2, .2, .3, .1],
                             [.2, .2, .3, .1],
                             [.2, .2, .3, .1],
                             [.2, .2, .3, .1]], dtype=np.float32)

        _w3_init = np.array([[.1, .8, .3, .1],
                             [.2, .4, .2, .3],
                             [.3, .6, .5, .1],
                             [.4, .2, .3, .7]], dtype=np.float32)

        b1 = np.zeros((n_hidden_1, ),dtype=np.float32)
        b2 = np.zeros((n_hidden_1,), dtype=np.float32)
        b3 = np.zeros((n_hidden_1,), dtype=np.float32)
        w_tensor1 = torch.tensor(_w1_init.T, requires_grad=True)
        w_tensor2 = torch.tensor(_w2_init.T, requires_grad=True)
        w_tensor3 = torch.tensor(_w3_init.T, requires_grad=True)
        b_tensor1 = torch.tensor(b1, requires_grad=True)
        b_tensor2 = torch.tensor(b2, requires_grad=True)
        b_tensor3 = torch.tensor(b3, requires_grad=True)

        self.l1.weight.data = torch.nn.Parameter(w_tensor1, requires_grad=True)
        self.l2.weight.data = torch.nn.Parameter(w_tensor2, requires_grad=True)
        self.l3.weight.data = torch.nn.Parameter(w_tensor3, requires_grad=True)
        self.l1.bias = torch.nn.Parameter(b_tensor1, requires_grad=True)
        self.l2.bias = torch.nn.Parameter(b_tensor2, requires_grad=True)
        self.l3.bias = torch.nn.Parameter(b_tensor3, requires_grad=True)
        #l2.bias.data = torch.Tensor(b1)

        #l3.bias.data = torch.Tensor(b1)

    def forward(self, x):
        #print(self.l1.weight.grad)
        #print(self.l1.bias.grad)
        #print(self.l2.weight.grad)
        #print(self.l2.bias.grad)
        print(self.l3.weight.grad)
        print(self.l3.bias.grad)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Batch_Net(nn.Module):
    """
    在上面的Activation_Net的基础上，增加了一个加快收敛速度的方法——批标准化
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.Tanh(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.Tanh(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    model = Activation_Net(6, 4, 4, 4)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    criterion = nn.CTCLoss()
    #criterion = nn.SmoothL1Loss()
    optimizer = optim.RMSprop(model.parameters(),lr=0.05,alpha=0.99,eps=1e-10)
    for e in range(100):
        x = np.array([.1, .2, .3, .4, .5, .6],dtype=np.float32)
        y = np.array([.4, .5, 0, .1],dtype=np.float32)



        x_tensor = torch.tensor(x, requires_grad=True)
        y_tensor = torch.tensor(y, requires_grad=True)
        out = model(x_tensor)
        #out.dim = 0
        #print(out.weight.grad.numpy())
        loss = criterion(out,y_tensor)#torch.argmax(label, dim=1)
        print("%.8f" % loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(str(e),'->',out)