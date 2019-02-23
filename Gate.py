import numpy as np

class Gate:
	def __init__(self,Name,T):
		self.Name = Name
		self.start_i = 0
		self.Gates = []
		self.T = T
	def setT(self,T):
		self.T = T
	def cleanGateTimes(self):
		self.Gates.clear()
		self.start_i = 0
	def runBackward(self):
		for i in range(self.start_i-1,0,-1):
			#print(i,self.start_i)
			self.Gates[i].backward()
		self.Gates.clear()
		self.start_i = 0
	def printGateTimes(self):
		for g in self.Gates:
			print(g)
	def updateGateTimes(self,g):
		self.Gates.append(g)
		self.start_i += 1


# str = 'NeuronGate(self).backward'
# self.P.updateGateTimes(str,activeFunc)

class AddGate(Gate):
	def __init__(self,P,Input1=None,Input2=None,o=None):
		self.P = P
		self.Input1 = 'self.T.' + Input1
		self.Input2 = 'self.T.' + Input2
		self.dz = 'self.T.d' + o
		self.T = self.P.T
		self.o = o
		self.doutput1 = 'd' + Input1
		self.doutput2 = 'd' + Input2
	def forward(self):
		setattr(self.T, self.o, eval(self.Input1 + '+' + self.Input2))
		self.P.updateGateTimes(self)

	def backward(self):
		doutput1 = eval(self.dz) * np.ones_like(eval(self.Input1))
		doutput2 = eval(self.dz) * np.ones_like(eval(self.Input2))
		setattr(self.T,self.doutput1,doutput1)
		setattr(self.T,self.doutput2,doutput2)


class NeuronGate(Gate):
	def __init__(self,P,Input=None,W=None,bias=None,o=None,activeFunc = None):
		self.P = P
		self.TZ = self.P.T
		self.Input = 'self.TZ.' + Input
		self.W = 'self.TZ.param.' + W
		self.bias = 'self.TZ.param.' + bias
		self.dz = 'self.TZ.d' + o
		self.o = o
		self.activeFunc = activeFunc
		self.dInput = 'd' + Input
		self.dw = 'd' + W
		self.dbias = 'd'+ bias
		self.upW = W
		self.upBias = bias

		self.w1 = np.zeros_like(eval(self.W))
		self.w2 = np.zeros_like(self.w1)
		self.w3 = np.zeros_like(self.w1)
		self.b1 = np.zeros_like(eval(self.bias))
		self.b2 = np.zeros_like(self.b1)
		self.b3 = np.zeros_like(self.b1)
		self.adam_t = 0
	def forward(self):
		i = eval(self.Input)
		w = eval(self.W).T
		b = eval(self.bias)
		i = self.TZ.Dropout(i,self.TZ.param.keepDropout)

		self._output = np.dot(w,i) + b#eval('np.dot('+ self.W + '.T,' + i + ') + ' + self.bias)
		if self.activeFunc is not None:
			self._output = self.activeFunc.forward(self._output)
		setattr(self.TZ,self.o,self._output)
		self.P.updateGateTimes(self)

	def backward(self):
		dz = eval(self.dz)
		if self.activeFunc is not None:
			dz = self.activeFunc.backward(self._output) * dz
		dw = np.asarray(np.dot(dz, eval(self.Input).T)).T
		dx = np.dot(eval(self.W), dz)
		dbias = dz
		setattr(self.TZ.param,self.dbias,dz)
		setattr(self.TZ.param,self.dw,dw)
		setattr(self.TZ,self.dInput,dx)

		self.update(dw,dbias)
		#return dw,dx
	def update(self,dw,dbias):
		self.w1, self.w2, self.w3,self.b1, self.b2,self.b3,self.adam_t, = self.TZ.param.Optimizer.Update(dw,self.upW,dbias,self.upBias,self.w1,self.w2,self.w3,self.b1,self.b2,self.b3,self.adam_t)


		#SGD
		#setattr(self.TZ.param,self.upW,eval(self.W)-rate* dw.T)
		#setattr(self.TZ.param, self.upBias, eval(self.bias) - rate * dbias)

		#SGD+Momentum
		#rho = 0.01
		#self.vx = rho * self.vx + dw
		#self.vb = rho * self.vb + dbias
		#setattr(self.TZ.param,self.upW,eval(self.W)-rate* self.vx.T)
		#setattr(self.TZ.param, self.upBias, eval(self.bias) - rate * self.vb)

		#AdaGrad
		#self.grad_squared_w += dw ** 2
		#self.grad_squared_b += dbias ** 2
		#setattr(self.TZ.param, self.upW, eval(self.W) - rate * dw.T / (np.sqrt(self.grad_squared_w.T)+ 1e-8))
		#setattr(self.TZ.param, self.upBias, eval(self.bias) - rate * dbias/(np.sqrt(self.grad_squared_b)+ 1e-8))


		#RMSProp
		#decay_rate = 0.9
		#dw = ddw.T
		#self.grad_squared_w = decay_rate * self.grad_squared_w + (1 - decay_rate) * dw ** 2
		#self.grad_squared_b = decay_rate * self.grad_squared_b + (1 - decay_rate) * dbias ** 2
		#setattr(self.TZ.param, self.upW, eval(self.W) - rate * dw / (np.sqrt(self.grad_squared_w) + 1e-8))
		#setattr(self.TZ.param, self.upBias, eval(self.bias) - rate * dbias / (np.sqrt(self.grad_squared_b) + 1e-8))

		#Adam
		#beta1 = 0.9
		#beta2 = 0.999
		#dw = dw.T
		#self.adam_t += 1
		#self.adam_m = beta1 * self.adam_m + (1 - beta1) * dw
		#self.adam_v = beta2 * self.adam_v + (1 - beta2) * (dw ** 2)
		#mb = self.adam_m / (1 - beta1 ** self.adam_t)
		#vb = self.adam_v / (1 - beta2 ** self.adam_t)
		#setattr(self.TZ.param, self.upW, eval(self.W) - rate * mb / (np.sqrt(vb) + 1e-8))

		#self.adam_bm = beta1 * self.adam_bm + (1 - beta1) * dbias
		#self.adam_bv = beta2 * self.adam_bv + (1 - beta2) * (dbias ** 2)
		#mb = self.adam_bm / (1 - beta1 ** self.adam_t)
		#vb = self.adam_bv / (1 - beta2 ** self.adam_t)
		#setattr(self.TZ.param, self.upBias, eval(self.bias) - rate * mb / (np.sqrt(vb) + 1e-8))





class MulGate(Gate):
	def __init__(self,P,Input1=None,Input2=None,o=None):
		self.P = P
		self.Input1 = 'self.T.' +Input1
		self.Input2 = 'self.T.' +Input2
		self.dz = 'self.T.d' + o
		self.o = o
		self.T = self.P.T

		self.dinput1 = 'd' + Input1
		self.dinput2 = 'd' + Input2
	def forward(self):
		setattr(self.T, self.o,np.multiply(eval(self.Input1),eval(self.Input2)))
		self.P.updateGateTimes(self)

	def backward(self):
		dinput1 = eval(self.dz) * eval(self.Input2)
		dinput2 = eval(self.dz) * eval(self.Input1)
		setattr(self.T,self.dinput1,dinput1)
		setattr(self.T, self.dinput2, dinput2)


class InOutGate(Gate):
	def __init__(self,P,Input=None,o=None,activeFunc = None):
		self.P = P
		self.T = self.P.T
		self.Input = 'self.T.' +Input
		self.dz = 'self.T.d' +o
		self.o = o
		self.activeFunc = activeFunc

		self.dInput = 'd' + Input
	def forward(self):
		if self.activeFunc is not None:
			self._output = self.activeFunc.forward(eval(self.Input))
		else:
			self._output = eval(self.Input)
		setattr(self.T,self.o,self._output)
		self.P.updateGateTimes(self)

	def backward(self):
		if self.activeFunc is not None:
			d = self.activeFunc.backward(self._output) * eval(self.dz)
		else:
			d = eval(self.dz)
		setattr(self.T,self.dInput,d)

class ConcateGate(Gate):
	def __init__(self, P,  Input1=None, Input2=None,o=None):
		self.P = P
		self.T = self.P.T
		self.Input1 = 'self.T.' + Input1
		self.Input2 = 'self.T.' + Input2
		self.dz = 'self.T.d' + o
		self.o = o

		self.doutput2 = 'd' + Input2
		self.doutput1 = 'd' + Input1

	def forward(self):
		setattr(self.T,self.o,np.concatenate((eval(self.Input1),eval(self.Input2)),axis=0))
		self.P.updateGateTimes(self)

	def backward(self):
		l = eval(self.Input1).shape[0]
		setattr(self.T, self.doutput1, eval(self.dz)[:l])
		setattr(self.T, self.doutput2,eval(self.dz)[l:])

		#return dz

class CopyGate(Gate):
	def __init__(self, P, Input=None, o=None):
		self.P = P
		self.T = self.P.T
		self.Input = 'self.T.' + Input
		self.dz = 'self.T.d' + o
		self.o = o
		self.doutput = 'd' + Input

	def forward(self):
		setattr(self.T, self.o, eval(self.Input))
		if hasattr(self.T, self.doutput) is True:#
			setattr(self.T, self.doutput, np.zeros_like(eval(self.dz)))
		self.P.updateGateTimes(self)
	def backward(self):

		if hasattr(self.T,self.doutput) is False:
			setattr(self.T, self.doutput, eval(self.dz))
		else:
			setattr(self.T, self.doutput, eval(self.dz) + eval('self.T.' + self.doutput))

class CNNGate(Gate):
	def __init__(self, P, Input=None,W=None,bias=None,o=None,activeFunc = None,fliters=3,step = 1,padding = 0,F_num = 1):
		self.P = P
		self.TZ = self.P.T
		self.Input = 'self.TZ.' + Input
		self.W = 'self.TZ.param.' + W
		self.bias = 'self.TZ.param.' + bias
		self.dz = 'self.TZ.d' + o
		self.o = o
		self.activeFunc = activeFunc
		self.dInput = 'd' + Input
		self.dw = 'd' + W
		self.dbias = 'd' + bias
		self.upW = W
		self.upBias = bias


		self.fliters = fliters
		self.step = step
		self.padding = padding
		self.FNum = F_num

		self.w1 = np.zeros_like(eval(self.W))
		self.w2 = np.zeros_like(self.w1)
		self.w3 = np.zeros_like(self.w1)
		self.b1 = np.zeros_like(eval(self.bias))
		self.b2 = np.zeros_like(self.b1)
		self.b3 = np.zeros_like(self.b1)
		self.adam_t = 0

	def getOutputSize(self, M, S, F, P):
		return (M - F + 2 * P) / S + 1

	def paddingZeros(self, input, P):
		if P > 0:
			p = np.zeros((input.shape[1], P))
			input = np.c_[input, p]
			input = np.c_[p, input]
			p = np.zeros((P, input.shape[0] + 2 * P))
			input = np.r_[input, p]
			input = np.r_[p, input]
		return input

	def conv(self, input, weight, outputW, outputH, step):
		result = np.zeros((outputH, outputW))
		fliter = weight.shape[0]
		for h in range(outputH):
			for w in range(outputW):
				i_a = input[h * step:h * step + fliter, w * step:w * step + fliter]
				result[h][w] = np.multiply(i_a,weight).sum()  # self.calc_connv(i_a,weight)#np.multiply(i_a,weight).sum()#self.calc_connv(i_a,weight)
		return result

	def forward(self):
		input = eval(self.Input)
		w = eval(self.W)
		b = eval(self.bias)

		inputW = input.shape[1]
		inputH = input.shape[0]
		self.inputW = inputW
		self.inputH = inputH
		outputW = int(self.getOutputSize(inputW, self.step, self.fliters, self.padding))
		outputH = int(self.getOutputSize(inputH, self.step, self.fliters, self.padding))
		self._output = np.zeros((self.FNum,outputW,outputH))
		input = self.paddingZeros(input, self.padding)

		for f in range(self.FNum):
			self._output[f] = self.conv(input, w[f], outputW, outputH, self.step)
			for i in range(outputW):
				for j in range(outputH):
					self._output[f][i][j] = self.activeFunc.forward(self._output[f][i][j] + b)

		setattr(self.TZ, self.o, self._output)
		self.P.updateGateTimes(self)
	def backward(self):
		input = eval(self.Input)
		dz = eval(self.dz)
		w = eval(self.W)

		dw = np.zeros_like(w)
		dx = np.zeros_like(input)
		for f in range(self.FNum):
			if self.activeFunc is not None:
				dz[f] = self.activeFunc.backward(self._output[f]) * dz[f]
			downdelta = dz[f]
			dw[f] = self.conv(input,downdelta,self.fliters,self.fliters,self.step)

			newDelta = self.paddingZeros(downdelta, 2)##############2?
			tmp_w = np.rot90(w[f], 2)
			p = self.conv(newDelta, tmp_w, self.inputW, self.inputH, self.step)
			a = np.zeros_like(input)
			for i in range(input.shape[0]):
				for j in range(input.shape[1]):
					a[i][j] = self.activeFunc.backward(input[i][j])
			dx = p * input
		dbias = dz.sum()
		setattr(self.TZ.param, self.dbias, dbias)
		setattr(self.TZ.param, self.dw, dw)
		setattr(self.TZ, self.dInput, dx)

		self.update(dw,dbias)
	def update(self, dw, dbias):
		self.w1, self.w2, self.w3, self.b1, self.b2, self.b3, self.adam_t, = self.TZ.param.Optimizer.Update(dw,
																												self.upW,
																												dbias,
																												self.upBias,
																												self.w1,
																												self.w2,
																												self.w3,
																												self.b1,
																												self.b2,
																												self.b3,
																												self.adam_t)

		#setattr(self.TZ.param, self.upW, eval('self.TZ.param.' + self.upW) - 0.05 * dw)
		#setattr(self.TZ.param, self.upBias, eval('self.TZ.param.' + self.upBias) - 0.05 * dbias)



class PoolGate(Gate):
	def __init__(self, P, Input=None,W=None,bias=None,o=None,activeFunc = None,fliters=2,step = 1,padding=0,F_num = 1,type='MAX'):
		self.P = P
		self.TZ = self.P.T
		self.Input = 'self.TZ.' + Input
		self.dz = 'self.TZ.d' + o
		self.o = o
		self.activeFunc = activeFunc
		self.dInput = 'd' + Input



		self.fliters = fliters
		#self.F_num = F_num
		self.step = step
		self.type = type

	def getOutputSize(self, M, S, F):
		return (M - F ) / S + 1

	def calc_pool(self, input, type, index, f=0):
		result = .0
		if type == 'MAX':
			max = input[0, 0]
			for i in range(input.shape[0]):
				for j in range(input.shape[1]):
					if input[i, j] > max:
						max = input[i, j]
						self.bz_x[f][index] = i
						self.bz_y[f][index] = j

			result = max
		elif type == 'AVERAGE':
			size = input.shape[0] + input.shape[1]
			for i in range(input.shape[0]):
				for j in range(input.shape[1]):
					result += input[i, j] / size
		return result


	def forward(self):
		input = eval(self.Input)

		self.FNum = input.shape[0]
		inputW = input.shape[2]
		inputH = input.shape[1]
		self.outputW = int(self.getOutputSize(inputW, self.step, self.fliters))
		self.outputH = int(self.getOutputSize(inputH, self.step, self.fliters))
		self._output = np.zeros((self.FNum, self.outputH, self.outputW))
		self.bz_x = np.zeros((self.FNum, self.outputW * self.outputH))
		self.bz_y = np.zeros((self.FNum, self.outputW * self.outputH))
		for f in range(self.FNum):
			index = 0
			for h in range(self.outputH):
				for w in range(self.outputW):
					i_a = input[f][h * self.step:h * self.step + self.fliters,
							w * self.step:w * self.step + self.fliters]
					self._output[f][h][w] = self.calc_pool(i_a, self.type, index, f)
					index += 1
		setattr(self.TZ, self.o, self._output)
		self.P.updateGateTimes(self)
	def backward(self):
		input = eval(self.Input)
		dz = eval(self.dz)
		result = np.zeros((input.shape))
		for f in range(self.FNum):
			if self.type == 'MAX':
				index = 0
				for i in range(self.outputW):
					for j in range(self.outputH):
						x = int(i * self.step + self.bz_x[f][index])
						y = int(j * self.step + self.bz_y[f][index])
						result[f][x][y] = dz[f][i][j]
						index += 1

			elif self.type == 'AVERAGE':
				for i in range(self.outputW):
					for j in range(self.outputH):
						pool_size = self.step * self.step
						for m in range(self.step):
							for n in range(self.step):
								result[f][i * self.step + m][j * self.step + n] = dz[f][i][j] / pool_size
		#result = result *
		setattr(self.TZ, self.dInput, result)


