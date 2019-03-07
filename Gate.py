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
			if 'SoftmaxActivator' == self.activeFunc.__class__.__name__  and self._output.shape[1] > 1:
				for n in range(self._output.shape[1]):
					self._output[:,n] = self.activeFunc.forward(self._output[:,n])
			else:
				self._output = self.activeFunc.forward(self._output)
		setattr(self.TZ,self.o,self._output)
		self.P.updateGateTimes(self)

	def backward(self):
		dz = eval(self.dz)
		if self.activeFunc is not None:
			if 'SoftmaxActivator' == self.activeFunc.__class__.__name__ and self._output.shape[1] > 1:
				for n in range(self._output.shape[1]):
					dz[:,n] = self.activeFunc.backward2(self._output[:,n],dz[:,n])
			else:
				dz = self.activeFunc.backward(dz)

		dw = np.asarray(np.dot(dz, eval(self.Input).T)).T
		dx = np.dot(eval(self.W), dz)
		if dz.shape[1] > 1:# N > 1
			b = eval(self.bias)
			dbias = np.zeros_like(b)
			for n in range(dz.shape[1]):
				tmp = dz[:,n]
				tmp = tmp.reshape(-1,1)
				dbias += tmp
		else:
			dbias = dz
		setattr(self.TZ.param,self.dbias,dbias)
		setattr(self.TZ.param,self.dw,dw)
		setattr(self.TZ,self.dInput,dx)
		#print('dw->',dw)
		self.update(dw,dbias)
		#return dw,dx
	def update(self,dw,dbias):
		self.w1, self.w2, self.w3,self.b1, self.b2,self.b3,self.adam_t, = self.TZ.param.Optimizer.Update(dw,self.upW,dbias,self.upBias,self.w1,self.w2,self.w3,self.b1,self.b2,self.b3,self.adam_t)

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

class FlattenGate(Gate):
	def __init__(self, P, Input=None, o=None):
		self.P = P
		self.T = self.P.T
		self.Input = 'self.T.' + Input
		self.dz = 'self.T.d' + o
		self.o = o
		self.doutput = 'd' + Input

		self.H = 0
		self.W = 0
		self.input_dim = 0
	def forward(self):
		self.input = eval(self.Input)
		l = 1
		for s in self.input.shape[1:]:
			l = l * s
		out = np.zeros((l,self.input.shape[0]))
		for n in range(self.input.shape[0]):
			out[:,n] = self.input[n].flatten()
		self._output = out
		setattr(self.T, self.o, self._output)
		self.P.updateGateTimes(self)

	def backward(self):
		dz = eval(self.dz)
		dinput = dz.reshape(self.input.shape)
		setattr(self.T, self.doutput, dinput)

class CNNGate(Gate):
	def __init__(self, P, Input=None,W=None,bias=None,o=None,activeFunc = None,fliters=3,step = 1,padding = 0,channel_in = 1,channel_out = 1):
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
		self.channel_in = channel_in
		self.channel_out = channel_out

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

	def expand_sensitivity_map_tride1(self, sensitivity_array):
		depth = sensitivity_array.shape[0]
		# 确定扩展后sensitivity map的大小
		# 计算stride为1时sensitivity map的大小
		expanded_width = (self.inputW -
						  self.fliters + 2 * self.padding + 1)
		expanded_height = (self.inputH -
						   self.fliters + 2 * self.padding + 1)
		# 构建新的sensitivity_map
		expand_array_tride1 = np.zeros((depth, expanded_height,
								 expanded_width))
		# 从原始sensitivity map拷贝误差值
		for i in range(self.output_height):
			for j in range(self.output_width):
				i_pos = i * self.stride
				j_pos = j * self.stride
				expand_array_tride1[:, i_pos, j_pos] = sensitivity_array[:, i, j]
		return expand_array_tride1

	def create_delta_array(self):
		return np.zeros_like(self.paddedInput)

	def forward(self):
		input = eval(self.Input)
		w = eval(self.W)
		b = eval(self.bias)
		inputW = input.shape[-1]
		inputH = input.shape[input.ndim - 1]
		self.inputW = inputW
		self.inputH = inputH
		outputW = int(self.getOutputSize(inputW, self.step, self.fliters, self.padding))
		outputH = int(self.getOutputSize(inputH, self.step, self.fliters, self.padding))
		self._output = np.zeros((self.channel_out, outputW, outputH))
		input = self.paddingZeros(input, self.padding)
		self.paddedInput = input

		nw = w#.transpose(3, 2, 0, 1)
		wn, wc, wh, ww = np.shape(nw)
		from Batch2ConvMatrix import Batch2ConvMatrix
		self.b2m = Batch2ConvMatrix(self.step, wh, ww)
		x2m = self.b2m(self.paddedInput)
		w2m = nw.reshape(wn, -1)
		xn, xc, oh, ow = self.b2m.conv_size
		out_matrix = np.matmul(x2m, w2m.T) + b
		out = out_matrix.reshape((xn, oh, ow, wn))
		self.x2m = x2m
		self.w2m = w2m
		out = out.transpose((0, 3, 1, 2))
		self._output = self.activeFunc.forward(out)
		setattr(self.TZ, self.o, self._output)
		self.P.updateGateTimes(self)
		return out

	def backward(self):
		dz = eval(self.dz)
		w = eval(self.W)
		dz = self.activeFunc.backward(dz)
		on, oc, oh, ow = np.shape(dz)
		dz = dz.transpose((0, 2, 3, 1))
		dz = dz.reshape((on * oh * ow, -1))
		dw = np.matmul(dz.T, self.x2m)
		dw = dw.reshape(np.shape(w))
		dbias = np.sum(dz, axis=0)
		dx2m = np.matmul(dz, self.w2m)
		dx = self.b2m.backward(dx2m)
		setattr(self.TZ.param, self.dbias, dbias)
		setattr(self.TZ.param, self.dw, dw)
		setattr(self.TZ, self.dInput, dx)
		self.update(dw, dbias)

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

	def calc_pool(self, input, type, index, f=0,c = 0):
		result = .0
		if type == 'MAX':
			max = input[0, 0]
			for i in range(input.shape[0]):
				for j in range(input.shape[1]):
					if input[i, j] > max:
						max = input[i, j]
						self.bz_x[f][c][index] = i
						self.bz_y[f][c][index] = j

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
		self.channel = input.shape[1]
		inputW = input.shape[3]
		inputH = input.shape[2]
		self.outputW = int(self.getOutputSize(inputW, self.step, self.fliters))
		self.outputH = int(self.getOutputSize(inputH, self.step, self.fliters))
		self._output = np.zeros((self.FNum, self.channel,self.outputH, self.outputW))
		self.bz_x = np.zeros((self.FNum, self.channel,self.outputW * self.outputH))
		self.bz_y = np.zeros((self.FNum, self.channel,self.outputW * self.outputH))
		for f in range(self.FNum):
			for c in range(self.channel):
				index = 0
				for h in range(self.outputH):
					for w in range(self.outputW):
						i_a = input[f][c][h * self.step:h * self.step + self.fliters,
								w * self.step:w * self.step + self.fliters]
						self._output[f][c][h][w] = self.calc_pool(i_a, self.type, index, f,c)
						index += 1
		setattr(self.TZ, self.o, self._output)
		self.P.updateGateTimes(self)
	def backward(self):
		input = eval(self.Input)
		dz = eval(self.dz)
		result = np.zeros((input.shape))
		for f in range(self.FNum):
			for c in range(self.channel):
				if self.type == 'MAX':
					index = 0
					for i in range(self.outputW):
						for j in range(self.outputH):
							x = int(i * self.step + self.bz_x[f][c][index])
							y = int(j * self.step + self.bz_y[f][c][index])
							result[f][c][x][y] = dz[f][c][i][j]
							index += 1

				elif self.type == 'AVERAGE':
					for i in range(self.outputW):
						for j in range(self.outputH):
							pool_size = self.step * self.step
							for m in range(self.step):
								for n in range(self.step):
									result[f][c][i * self.step + m][j * self.step + n] = dz[f][c][i][j] / pool_size
		#result = result *
		setattr(self.TZ, self.dInput, result)
