#!/usr/bin/env Python
# coding=utf-8
from ActiveFunc import TanhActivator,softmax3,SigmoidActivator
import numpy as np
from Gate import NeuronGate,AddGate,MulGate,InOutGate,ConcateGate,CopyGate,Gate
from Json2Network import JsonModel


class Param:
	def __init__(self,isize,hsize,osize,rate=0.1):
		self.inputsize = isize
		self.outputsize = osize
		self.hiddensize = hsize
		self.hsize = hsize
		self.rate=rate
		self.ModelFile = './gojs/model2.txt'
class bp:
	def __init__(self,param):
		self.param = param
		self.jsonModel = JsonModel(self.param)
		self.jsonModel.run()
		self.Gate = Gate('', self)
		self.Gates = self.jsonModel.getGatesAndParam(self)


	def forward(self, input):

		setattr(self,self.jsonModel.input,input)

		for db in self.jsonModel.DB:  # fromDB
			key = 'o' + self.jsonModel.key2bz(db['key'])
			value = 'self.' + db['text']
			setattr(self, key, eval(value))

		for g in self.Gates:  # forward
			str = 'self.' + g.Key + '.forward()'
			eval(str)

		for db in self.jsonModel.Output:  # toDB
			value = 'self.o' + self.jsonModel.key2bz(db['key'])
			key = db['text']
			setattr(self, key, eval(value))

	def backward(self):
		for output in self.jsonModel.Output:  # fromDB
			key = 'do' + self.jsonModel.key2bz(output['key'])
			value = 'self.d' + output['text']
			setattr(self, key, eval(value))

		self.Gate.runBackward()  # backward

		for db in self.jsonModel.DB:  # toDB
			value = 'self.do' + self.jsonModel.key2bz(db['key'])
			key = 'd' + db['text']
			setattr(self, key, eval(value))

	def setOutputDelta(self, target,Y):
		delta = (target - Y)
		self.dOutput = delta

if __name__ == '__main__':

	x = np.array([[1], [2], [3]])
	y = np.array([[.1], [.8], [.3]])

	param = Param(3,5,3)
	network = bp(param=param)

	for e in range(1000):
		network.forward(x)
		t = network.Output
		print(t)
		network.setOutputDelta(t,y)
		network.backward()