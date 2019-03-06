import numpy as np

class SGDOptimizer:#ok
	def __init__(self,param):
		self.param = param
	def Update(self,dw,upW,dbias,upBias,w1=None,w2=None,w3=None,b1=None,b2=None,b3=None,t=None):
		rate = eval('self.param.rate')
		setattr(self.param,upW,eval('self.param.'+upW)-rate* dw)
		setattr(self.param, upBias, eval('self.param.'+upBias) - rate * dbias)
		return w1, w2,w3, b1, b2,b3, t
class SGDandMomentumOptimizer:#ok
	def __init__(self, param):
		self.param = param
	def Update(self, dw, upW, dbias, upBias,w1=None,w2=None,w3=None,b1=None,b2=None,b3=None,t=None):
		rate = eval('self.param.rate')
		rho = 0.9
		w1 = rho * w1 + dw
		b1 = rho * b1 + dbias
		setattr(self.param,upW,eval('self.param.' + upW)-rate* w1)
		setattr(self.param, upBias, eval('self.param.' + upBias) - rate * b1)
		return w1, w2,w3, b1, b2,b3, t
class SGDNesterovMomentumOptimizer:#ok
	def __init__(self, param):
		self.param = param

	def Update(self, dw, upW, dbias, upBias, w1=None, w2=None, w3=None,b1=None,b2=None,b3=None,t=None):
		rate = eval('self.param.rate')
		rate = 0.001
		mu = 0.9
		w1 = mu * w1 - rate * dw
		w1 = mu * w1 - rate * dw

		b1 = mu * b1 - rate * dbias
		b1 = mu*b1 - rate*dbias

		'''
		prev_w1 = w1
		w1 = mu * w1 - rate * dw
		w1 = w1 + mu * (w1 - prev_w1)
		##w1 = (1+mu) * w1 - mu * prev_w1

		prev_b1 = b1
		b1 = mu * b1 - rate * dbias
		b1 = b1 + mu * (b1 - prev_b1)
		'''
		setattr(self.param, upW, eval('self.param.' + upW) + w1)
		setattr(self.param, upBias, eval('self.param.' + upBias) +b1)
		return w1, w2,w3, b1, b2,b3, t
class AdaGradOptimizer:#ok
	def __init__(self, param):
		self.param = param

	def Update(self, dw, upW, dbias, upBias, w1=None, w2=None, w3=None,b1=None,b2=None,b3=None,t=None):
		rate = eval('self.param.rate')
		initial_accumulator_value = 0.1
		if ((w1 == 0).any()):
			w1 = w1 + initial_accumulator_value
		if ((b1 == 0).any()):
			b1 = b1 + initial_accumulator_value
		w1 += (dw ** 2)
		b1 += (dbias ** 2)
		setattr(self.param, upW, eval('self.param.' + upW) - rate * dw / (np.sqrt(w1) + 1e-10))
		setattr(self.param, upBias, eval('self.param.' + upBias) - rate * dbias / (np.sqrt(b1) + 1e-10))
		return w1, w2,w3, b1, b2,b3, t

class RMSPropOptimizer:#ok
	def __init__(self, param):
		self.param = param

	def Update(self, dw, upW, dbias, upBias, w1=None, w2=None, w3=None,b1=None,b2=None,b3=None,t=None):
		rate = eval('self.param.rate')
		decay_rate = 0.99
		w1 = decay_rate * w1 + (1 - decay_rate) * np.square(dw)
		b1 = decay_rate * b1 + (1 - decay_rate) * np.square(dbias)

		setattr(self.param, upW, eval('self.param.' + upW) - rate * dw / (np.sqrt(w1)+ 1e-10))
		setattr(self.param, upBias, eval('self.param.' + upBias) - rate * dbias / (np.sqrt(b1) + 1e-10))
		return w1, w2,w3, b1, b2,b3, t

class RMSPropWithNesterovMomentumOptimizer:
	def __init__(self, param):
		self.param = param

	def Update(self, dw, upW, dbias, upBias, w1=None, w2=None, w3=None,b1=None,b2=None,b3=None,t=None):
		rate = eval('self.param.rate')
		decay_rate = 0.9
		rho = 0.9
		w1 = decay_rate * w1 + (1 - decay_rate) * (dw ** 2)
		w2 = rho * w2 - rate * dw / (np.sqrt(w1) + 1e-10)
		b1 = decay_rate * b1 + (1 - decay_rate) * (dbias ** 2)
		b2 = rho*b2 - rate * dbias / (np.sqrt(b1) + 1e-10)

		setattr(self.param, upW, eval('self.param.' + upW) + w2)
		setattr(self.param, upBias, eval('self.param.' + upBias) + b2)
		return w1, w2,w3, b1, b2,b3, t

class AdamOptimizer:#ok
	def __init__(self, param):
		self.param = param

	def Update(self, dw, upW, dbias, upBias, w1=None, w2=None, w3=None,b1=None,b2=None,b3=None,t=None):
		rate = eval('self.param.rate')
		beta1 = 0.9
		beta2 = 0.999
		t += 1
		w1 = beta1 * w1 + (1 - beta1) * dw
		w2 = beta2 * w2 + (1 - beta2) * (dw ** 2)
		mb = w1 / (1 - beta1 ** t)
		vb = w2 / (1 - beta2 ** t)
		# setattr(self.TZ.param, self.upW, eval(self.W) - rate * mb / (np.sqrt(vb) + 1e-8))

		b1 = beta1 * b1 + (1 - beta1) * dbias
		b2 = beta2 * b2 + (1 - beta2) * (dbias ** 2)
		mb2 = b1 / (1 - beta1 ** t)
		vb2 = b2 / (1 - beta2 ** t)
		# setattr(self.TZ.param, self.upBias, eval(self.bias) - rate * mb / (np.sqrt(vb) + 1e-8))

		setattr(self.param, upW, eval('self.param.' + upW) - rate * mb / (np.sqrt(vb)+ 1e-8))
		setattr(self.param, upBias, eval('self.param.' + upBias) - rate * mb2 / (np.sqrt(vb2)+ 1e-8))
		return w1, w2,w3, b1, b2,b3, t


class AdadeltaOptimizer:#ok
	def __init__(self, param):
		self.param = param

	def Update(self, dw, upW, dbias, upBias, w1=None, w2=None, w3=None,b1=None,b2=None,b3=None,t=None):
		rate = eval('self.param.rate')
		rate = 0.001
		rho = 0.95
		epsilon = 1e-10
		w1 = rho * w1 + (1-rho) * (dw**2)
		w2 = np.sqrt((w3 + epsilon)/(w1 + epsilon)) * dw

		b1 = rho * b1 + (1 - rho) * (dbias ** 2)
		b2 = np.sqrt((b3 + epsilon) / (b1 + epsilon)) * dbias
		setattr(self.param, upW, eval('self.param.' + upW) - rate * w2)
		setattr(self.param, upBias, eval('self.param.' + upBias) - rate * b2)
		w3 = rho * w3 + (1-rho) * (w2 ** 2)
		b3 = rho * b3 + (1 - rho) * (b2 ** 2)


		return w1, w2,w3, b1, b2,b3, t