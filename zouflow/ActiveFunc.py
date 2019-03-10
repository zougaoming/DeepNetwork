# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
class ReluActivator(object):
	def forward(self, weighted_input): # 前向计算，计算输出
		#self._output = max(0, weighted_input)
		self._output = np.maximum(weighted_input,0)
		return self._output
	def backward(self, dz): # 后向计算，计算导数
		dnx = np.where(self._output>0,dz,0)
		#dnx = (1 if self._output > 0 else 0) * dz
		return dnx

# IdentityActivator激活器.f(x)=x
class IdentityActivator(object):
	def forward(self, weighted_input):
		self._output = weighted_input
		return self._output
	def backward(self, dz):
		# 后向计算，计算导数
		return dz

#Sigmoid激活器
class SigmoidActivator(object):
	def forward(self, weighted_input):
		self._output = 1.0 / (1.0 + np.exp(-weighted_input))
		return self._output
	def backward(self, dz):
		dnx = np.multiply(self._output, (1 - self._output)) * dz
		return dnx

# tanh激活器
class TanhActivator(object):
	def forward(self, weighted_input):
		self._output = 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0
		return self._output
	def backward(self, dz):
		dnx = (1 - self._output * self._output) * dz
		return dnx
	def backward2(self,output):
		return (1 - output * output)

class SoftmaxActivator:#ok
	def forward(self, nx):
		shifted_x = nx - np.max(nx)
		ex = np.exp(shifted_x)
		sum_ex = np.sum(ex)
		softmax = ex / sum_ex
		self._output = softmax.copy()
		return self._output
	def backward(self, dz):
		grad = self._output[:, np.newaxis] * self._output[np.newaxis, :]
		for i in range(len(grad)):
			grad[i, i] -= self._output[i]
		grad = -grad
		dnx = np.sum(grad * dz,axis=1)
		return dnx

	def backward2(self, output,dz):
		grad = output[:, np.newaxis] * output[np.newaxis, :]
		for i in range(len(grad)):
			grad[i, i] -= output[i]
		grad = -grad
		dnx = np.sum(grad * dz, axis=1)
		return dnx

class LogSoftmaxActivator:
	def forward(self, nx):#ok
		self.nx = nx
		shifted_x = nx - np.max(nx)
		ex = np.exp(shifted_x)
		sum_ex = np.sum(ex)
		self.s = ex / sum_ex
		self._output = np.log(self.s.copy())
		return self._output
	def backward(self, dz):#ok
		target = self.s - dz
		self.dnx = self.s * np.sum(target)
		self.dnx -= target
		return self.dnx