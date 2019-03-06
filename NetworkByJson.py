from ActiveFunc import *
from Gate import *
from Loss import *
from Json2Network import JsonModel
from Optimizer import *
import numpy as np

class Param:
	def __init__(self,ModelFile,inputsize=0,outputsize=0,keepDropout=0,Optimizer=None,rate=0.05,ActiveFunc=None):
		self.ModelFile = ModelFile
		self.keepDropout = keepDropout
		self.inputsize = inputsize
		self.outputsize = outputsize
		self.rate = rate
		self.ActiveFunc = ActiveFunc
		if Optimizer is None:
			self.Optimizer = SGDOptimizer(self)
		else:
			self.Optimizer = Optimizer
class Network:
	def __init__(self, param):
		self.param = param
		self.jsonModel = JsonModel(self.param)
		self.jsonModel.run()
		self.Gate = Gate('', self)
		self.Gates = self.jsonModel.getGatesAndParam(self)

	def forward(self, input):

		setattr(self, self.jsonModel.input, input)
		for db in self.jsonModel.DB:  # fromDB
			key = 'o' + self.jsonModel.key2bz(db['key'])
			value = 'self.' + db['text']
			setattr(self, key, eval(value))

		for g in self.Gates:  # forward
			str = 'self.' + g.Key + '.forward()'
			print(str)
			eval(str)

		result = []
		for db in self.jsonModel.Output:  # toDB
			value = 'self.o' + self.jsonModel.key2bz(db['key'])
			key = db['text']
			k = key,eval(value)
			result.append(k)
			setattr(self, key, eval(value))

		return result

	def backward(self):
		for output in self.jsonModel.Output:  # fromDB
			key = 'do' + self.jsonModel.key2bz(output['key'])
			value = 'self.d' + output['text']
			setattr(self, key, eval(value))

		#self.Gate.printGateTimes()
		self.Gate.runBackward()  # backward
		result = []
		for db in self.jsonModel.DB:  # toDB
			value = 'self.do' + self.jsonModel.key2bz(db['key'])
			key = 'd' + db['text']
			k = key, eval(value)
			result.append(k)
			setattr(self, key, eval(value))
		return result

	def setOutputDelta(self, target):
		#m = target.shape[0]
		#delta = (-target + self.Output)
		Loss = SquareLoss()
		#print(Loss.loss(self.Output,target))
		delta = Loss.backward(self.Output,target)
		#delta = softmax3Activator().loss(target,self.Output)
		self.dOutput = delta

	# dropout函数的实现
	def Dropout(self,x, level):
		if level == 0:
			return x
		if level < 0. or level >= 1:  # level是概率值，必须在0~1之间
			raise ValueError('Dropout level must be in interval [0, 1[.')
		retain_prob = 1. - level

		# 我们通过binomial函数，生成与x一样的维数向量。binomial函数就像抛硬币一样，我们可以把每个神经元当做抛硬币一样
		# 硬币 正面的概率为p，n表示每个神经元试验的次数
		# 因为我们每个神经元只需要抛一次就可以了所以n=1，size参数是我们有多少个硬币。
		random_tensor = np.random.binomial(n=1, p=retain_prob,
										   size=x.shape)  # 即将生成一个0、1分布的向量，0表示这个神经元被屏蔽，不工作了，也就是dropout了
		x *= random_tensor
		#x /= retain_prob
		np.true_divide(x,retain_prob,out=x, casting='unsafe')
		return x