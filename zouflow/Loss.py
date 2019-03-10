import numpy as np
import math
class Loss:
	def where(self,condition,value,other_value):
		return np.where(condition,value,other_value)

class MSELoss:#ok
	def loss(self, output, target):
		self.output = output
		self.target = target
		return np.sum(np.square(self.output - self.target)) / self.output.size
	def backward(self,output, target):
		dx = 2 * (output - target) / output.size
		return dx
class SquareLoss:#ok
	def loss(self, output, target):
		self.output = output
		self.target = target
		return np.square(self.output - self.target)
	def backward(self, output, target):
		dx = 2 * (output - target)
		return dx
class SmoothL1Loss(Loss):#err
	def loss(self, output, target):
		self.output = output
		self.target = target
		t = np.abs(output - target)
		return np.mean(np.where(t < 1, 0.5 * t ** 2, t - 0.5))
	def backward(self, output, target):
		t = (output - target)
		dnx = np.where(t<1, t,1)
		#print(np.mean(dx))
		#dx = 0.5 * (output - target) ** 2
		return dnx

class EntropyLoss:#err
	def loss(self,output,target):
		self.output = output
		self.target = target
		loss = np.sum(- target * np.log(output))
		return loss
	def backward(self,output,target):
		dnx = - target / output
		return dnx

class SolfmaxEntropyLoss:#ok softmax_cross_entropy_with_logits
	def __init__(self):
		self.nx = None
		self.ny = None
		self.softmax = None
		self.entropy = None
		self.lossv = None
		self.dnx = None
	def loss(self, nx, ny):
		self.nx = nx
		self.ny = ny
		shifted_x = nx - np.max(nx)
		ex = np.exp(shifted_x)
		sum_ex = np.sum(ex)
		self.softmax = ex / sum_ex
		self.entropy = - np.log(self.softmax) * ny
		self.lossv = np.sum(self.entropy)
		return self.lossv
	def backward(self,output,target):
		self.dnx = output * np.sum(target)
		self.dnx -= target
		return self.dnx
