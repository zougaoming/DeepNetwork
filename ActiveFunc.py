# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
class ReluActivator(object):
	def forward(self, weighted_input): # 前向计算，计算输出
		return max(0, weighted_input)
	def backward(self, output): # 后向计算，计算导数
		return 1 if output > 0 else 0

# IdentityActivator激活器.f(x)=x
class IdentityActivator(object):
	def forward(self, weighted_input):
		# 前向计算，计算输出
		return weighted_input
	def backward(self, output):
		# 后向计算，计算导数
		return 1

#Sigmoid激活器
class SigmoidActivator(object):
	def forward(self, weighted_input):
		return 1.0 / (1.0 + np.exp(-weighted_input))
	def backward(self, output):
		# return output * (1 - output)
		return np.multiply(output, (1 - output)) # 对应元素相乘

# tanh激活器
class TanhActivator(object):
	def forward(self, weighted_input):
		return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0
	def backward(self, output):
		return 1 - output * output


class SoftmaxActivator(object):
	def forward(self,x):
		"""
		Compute the softmax function for each row of the input x.

		Arguments:
		x -- A N dimensional vector or M x N dimensional numpy matrix.

		Return:
		x -- You are allowed to modify x in-place
		"""
		orig_shape = x.shape

		if len(x.shape) > 1:
			# Matrix
			exp_minmax = lambda x: np.exp(x - np.max(x))
			denom = lambda x: 1.0 / np.sum(x)
			x = np.apply_along_axis(exp_minmax, 1, x)
			denominator = np.apply_along_axis(denom, 1, x)

			if len(denominator.shape) == 1:
				denominator = denominator.reshape((denominator.shape[0], 1))

			x = x * denominator
		else:
			# Vector
			x_max = np.max(x)
			x = x - x_max
			numerator = np.exp(x)
			denominator = 1.0 / np.sum(numerator)
			x = numerator.dot(denominator)

		assert x.shape == orig_shape
		return x
	def backward(self,output):
		N = output.shape[0]
		#dscores = self.top_val.copy()
		#dscores[range(N), list(output)] -= 1  # loss对softmax层的求导
		output /= N
		return output



class Softmax2:
	def __init__(self):
		self.softmax = None
		self.grad = None
		self.dnx = None
	def __call__(self, nx):
		shifted_x = nx - np.max(nx)
		ex = np.exp(shifted_x)
		sum_ex = np.sum(ex)
		self.softmax = ex / sum_ex
		return self.softmax
	def get_grad(self):
		self.grad = self.softmax[:, np.newaxis] * self.softmax[np.newaxis, :]
		for i in range(len(self.grad)):
			self.grad[i, i] -= self.solfmax[i]
			self.grad = - self.grad
		return self.grad
	def backward(self, dl):
		self.get_grad()
		self.dnx = np.sum(self.grad * dl, axis=1)
		return self.dnx


class softmax3:
	def forward(self, x):
		return np.exp(x) / np.sum(np.exp(x))

	def loss(self, x, y):
		probs = self.forward(x)
		return -np.log(probs[0, y])

	def backward(self, x, y):
		return x - y