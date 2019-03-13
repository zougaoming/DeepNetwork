
class GetSet(object):
	def __init__(self,network):
		self.network = network
		self.param = network.param
		self.Gates = network.Gates

	def getGate(self,otherKey):
		for g in self.Gates:
			if g.otherKey == otherKey:
				return g
		return None
	def _Get(self,_bz,_class,otherKey):
		g = self.getGate(otherKey)
		if g is None: return None
		s = _bz + '_'  +str(abs(g.t))
		if hasattr(eval('self.' + _class), s):
			return eval('self.' + _class + '.' + s)
		else:
			return None

	def _Set(self,_bz,_class,otherKey,Value):
		g = self.getGate(otherKey)
		if g is None: return False
		s = _bz + '_' + str(abs(g.t))
		setattr(eval('self.' + _class),s,Value)
		return True
	def getW(self,otherKey):
		return self._Get('w','param',otherKey)

	def setW(self,otherKey,W):
		return  self._Set('w','param',otherKey,W)

	def getBias(self,otherKey):
		return self._Get('b', 'param', otherKey)

	def setBias(self, otherKey, b):
		return self._Set('b', 'param', otherKey, b)

	def getOutput(self,otherKey):
		return self._Get('o', 'network', otherKey)


	def getGard(self, otherKey):
		return self._Get('d', 'param', otherKey)


