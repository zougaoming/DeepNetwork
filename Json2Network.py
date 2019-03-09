#!/usr/bin/env Python
# coding=utf-8
import json
from Gate import *
from Optimizer import *
from ActiveFunc import *
import numpy as np
class Link:
	def __init__(self,f,t,f2=0,Key=None,Value = None,inputsize=0,outputsize=0):
		self.f = f
		self.t = t
		self.Key = Key
		self.Value = Value
		self.f2 = f2
		self.inputsize = inputsize#f_num
		self.outputsize = outputsize#flite
		self.channel_in = 0
		self.channel_out = 0
class JsonModel:
	def __init__(self,param=None):
		self.param = param
		str = self.readJsonModelFile()
		self.jsonData = json.loads(str)
		self.jsonNode = self.jsonData['nodeDataArray']
		self.jsonLink = self.jsonData['linkDataArray']
		self.GatesLinked = []
		self.Links = []
		self.Gates = []
		self.WS = []
		self.BiasS = []
		self.DB = []
		self.upDB = []
		self.Output = []
		self.input = ''

	def readJsonModelFile(self):
		file_object = open(self.param.ModelFile)
		try:
			file_context = file_object.read()
		finally:
			file_object.close()
		if file_context.startswith(u'\ufeff'):
			file_context = file_context.encode('utf8')[3:].decode('utf8')
		return file_context
	def run(self):
		self.productLinker()
		self.initParam()
		self.productLayer()
		for i in range(3):
			self.orderGates()

		for i in range(1):
			self.setInputSizeAndOutputSize()



	def getGatesAndParam(self,T):

		#self.Gates[-1].outputsize = self.param.outputsize
		'''


		index = 0
		len_g = len(self.Gates)

		for g in self.Gates:
			if index + 1 < len_g:
				gn = self.Gates[index + 1]
				if int(g.outputsize) > 0:
					gn.inputsize = g.outputsize
				elif int(gn.inputsize) > 0:
					g.outputsize = gn.inputsize
			index += 1
		for g in self.Gates:
			if ('neuron' not in g.Key) and ('concate' not in g.Key):
				if int(g.outputsize) > 0:
					g.inputsize = g.outputsize
				elif int(g.inputsize) > 0:
					g.outputsize = g.inputsize
		#for g in self.Gates:
			#print(g.Value, g.inputsize, g.outputsize)
		'''
		self.WS_KV = []
		self.BiasS_KV = []
		for g in self.Gates:
			if ('neuron' in g.Key):
				for ws in self.WS:
					if '"' + ws + '"' in g.Value:
						value = 'np.random.uniform(-0.01, 0.01, size=('+ str(g.inputsize) +', '+ str(g.outputsize) +'))'
						key = ws
						#print(key,value)
						setattr(self.param,key,eval(value))
						break
				for bias in self.BiasS:
					if '"' + bias + '"' in g.Value:
						#value = 'np.random.uniform(-0.01, 0.01, size=(' + str(g.outputsize) + ', 1))'
						value = 'np.zeros((' + str(g.outputsize) + ',1 ))'
						key = bias
						setattr(self.param, key, eval(value))
						break
			if('cnn' in g.Key):
				for ws in self.WS:
					if '"' + ws + '"' in g.Value:
						value = 'np.random.uniform(-0.01, 0.01, size=(' + str(g.channel_out) + ', ' + str(
							g.channel_in) + ',' + str(g.outputsize) + ',' + str(g.outputsize) + '))'
						key = ws
						# print(key,value)
						setattr(self.param, key, eval(value))
						break
				for bias in self.BiasS:
					if '"' + bias + '"' in g.Value:
						#value = 'np.random.uniform(-0.01, 0.01, size=(' + str(g.channel_out) + ', 1))'
						value = 'np.zeros((' + str(g.channel_out) + ', ))'
						#value = 0
						key = bias
						setattr(self.param, key, eval(value))
						break

		for db in self.DB:
			g = self.findInputGateBykey(db['key'])
			setattr(T, db['text'], eval('np.zeros(('+ str(g.outputsize) +',1))'))
			#setattr(T, db['text'], eval('np.zeros((2,1))'))
			#print(db['text'],str(g.outputsize))

		for o in self.Output:
			g = self.findOutputGateBykey(o['key'])
			setattr(T, 'd' + o['text'], eval('np.zeros(('+ str(g.outputsize) +',1))'))
			#print(T.ds)
			#print(T.dOutput)
		for g in self.Gates:
			#print(g.Value)
			setattr(T, g.Key, eval(g.Value))

		return self.Gates

	def findInputGateBykey(self,key):
		for g in self.Gates:
			if g.f == key or g.f2 == key:
				return g

	def findOutputGateBykey(self, key):
		for g in self.Gates:
			if g.t == key:
				return g
	def orderGates(self):
		t = True
		index = 0
		while t == True:
			t = self.checkInputError(index)
			index += 1
	def checkInputError(self,index):
		t = 0
		for g in self.Gates[index:]:
			f1 = g.f
			f2 = g.f2
			t +=1
			for g in self.Gates[index+ t:]:
				if g.t == f1 or g.t == f2:
					self.Gates.remove(g)
					self.Gates.insert(index+t-1, g)
					return True
		return False
	def setInputSizeAndOutputSize(self):
		index = 0
		for g in self.Gates:
			if ('neuron' not in g.Key) and ('concate' not in g.Key) and ('cnn' not in g.Key):
				if int(g.outputsize) > 0:
					g.inputsize = g.outputsize
				if int(g.inputsize) > 0:
					g.outputsize = g.inputsize
			index += 1
			if(index  < len(self.Gates)):
				for gn in self.Gates[index:]:
					if gn.f == g.t or gn.f2 == g.t:
						if ('neuron' not in gn.Key) and ('concate' not in gn.Key) and ('cnn' not in g.Key):
							if int(g.outputsize) > 0:
								gn.inputsize = g.outputsize
							if int(gn.inputsize) > 0:
								g.outputsize = gn.inputsize
						if ('neuron' not in g.Key) and ('concate' not in g.Key) and ('cnn' not in g.Key):
							if int(g.outputsize) > 0:
								g.inputsize = g.outputsize
							if int(g.inputsize) > 0:
								g.outputsize = g.inputsize
		'''
		index = 0
		for g in self.Gates:
			index += 1
			if ('neuron' in g.Key) and ('concate' in g.Key):
				continue
			#print('curkey->', g.f,g.f2)
			if (index < len(self.Gates)):
				for gn in self.Gates[:index]:
					if ('neuron' in gn.Key) and ('concate' in gn.Key):
						continue
					#print('pp->',gn.t)
					if gn.t == g.f or gn.t == g.f2:
						#print('yes')
						if int(g.inputsize) > 0:
							gn.outputsize = g.inputsize

		'''


		#for g in self.Gates:
			#print(g.Value,g.inputsize,g.outputsize)

	def setInputSizeAndOutputSize2(self,index):
		t = 0
		for g in self.Gates[index:]:
			to = g.t
			t += 1

			for g2 in self.Gates[index + t:]:
				if g2.f == to or g2.f2 == to:
					if int(g.outputsize) > 0:
						g2.inputsize = g.outputsize
					return True
		return False

	def getInputNodeKey(self, key,needSord=False):
		Keys = []
		bkey = 0
		for link in self.Links:
			if link.t == key:
				node = self.Key2Node(link.f)
				if ('figure' in node) and node['figure'] == 'Value':
					continue
				Keys.append(link.f)
				bkey = link.f
		if needSord:
			Keys2 = []
			key = Keys[0]
			Keyst = self.getInputNodeKey(key)
			for key2 in Keyst:
				node = self.Key2Node(key2)
				if node['figure'] == 'Value':
					if 'after' in node['text']:
						Keys2.append(bkey)
						Keys2.append(key)
						break
					else:
						Keys2.append(key)
						Keys2.append(bkey)

			if len(Keys2) == 2:
				return Keys2
		return Keys
	def getOutputNodeKey(self,key):
		Keys = []
		for link in self.Links:
			if link.f == key or link.f2 == key:
				Keys.append(link.f)
		return Keys

	def Key2Node(self, key):
		for node in self.jsonNode:
			if node['key'] == key:
				return node
	def key2bz(self, key):
		return str(key).replace('-', '_')

	def getValue(self,key):
		Nodes = []
		for link in self.Links:
			if link.t == key:
				node = self.Key2Node(link.f)
				if ('figure' in node) and node['figure'] == 'Value':
					Nodes.append(node)
		return Nodes

	def productLinker(self):
		for linkData in self.jsonLink:
			link = Link(linkData['from'], linkData['to'])
			self.Links.append(link)

	def initParam(self):
		for node in self.jsonNode:
			text = node['text']
			key = node['key']
			if ('figure' in node) and node['figure'] == 'Value':
				if len(self.getInputNodeKey(key)) == 0 and len(self.getOutputNodeKey(key)) == 0:
					if 'I:' == text[0:2]:
						#self.inputsize = text[2:]
						setattr(self.param,'inputsize',int(text[2:]))
					elif 'O:' == text[0:2]:
						setattr(self.param, 'outputsize', int(text[2:]))
						#self.outputsize = text[2:]
					elif 'IO' == text[0:2]:
						tmp = text.split(':', 2)
						setattr(self.param, 'w_input', int(tmp[1]))
						setattr(self.param, 'w_output', int(tmp[2]))
					elif 'Rate:' == text[0:5]:
						setattr(self.param, 'rate', float(text[5:]))
					elif 'Activator' in text:
						setattr(self.param, 'ActiveFunc', text + '()')
					elif 'Optimizer' in text:
						#setattr(self.param, 'Optimizer', eval('AdadeltaOptimizer(self.param)'))
						setattr(self.param, 'Optimizer', eval(text + '(self.param)'))
					elif 'Dropout:' == text[0:8]:
						setattr(self.param, 'keepDropout', float(text[8:]))
					elif 'Loss:' in text:
						setattr(self.param, 'Loss', eval(text + '()'))
	def productLayer(self):
		for node in self.jsonNode:
			text = node['text']
			key = node['key']
			bz = self.key2bz(key)
			o = 'o' + bz
			#print(node)
			if text.lower() == 'input':
				s = 'InOutGate(T.Gate,Input="input",o="' + o + '")'
				l = Link(0, key, 0, 'inputGate' + bz,s,inputsize=self.param.inputsize,outputsize=self.param.inputsize)
				self.Gates.append(l)
				self.input = 'input'
			elif text.lower() == 'neuron':
				w = 'w' + bz
				bias = 'b' + bz
				upNodes = self.getInputNodeKey(key)
				Input = 'o' + self.key2bz(upNodes[0])

				Values = self.getValue(key)
				activeFunc = None
				outputsize = 0
				inputsize = 0
				for v in Values:
					if 'Activator' in v['text']:
						activeFunc = v['text'] + '()'
					elif 'IO' in v['text']:
						tmp = v['text'].split(':',2)
						inputsize = int(tmp[1])
						outputsize = int(tmp[2])
				if activeFunc == None and hasattr(self.param,'ActiveFunc'):
					activeFunc = self.param.ActiveFunc
				if inputsize == 0:
					inputsize = self.param.w_input
				if outputsize == 0:
					outputsize = self.param.w_output
				s = 'NeuronGate(T.Gate,Input="' + Input + '",W="' + w + '",bias="' + bias + '",o="' + o + '",activeFunc='+ activeFunc +')'
				l = Link(upNodes[0], key, 0, 'neuronGate' + bz,s,inputsize=inputsize,outputsize=outputsize)
				self.Gates.append(l)
				self.WS.append(w)
				self.BiasS.append(bias)
			elif text.lower() == 'concate':
				upNodes = self.getInputNodeKey(key,needSord=True)
				Input1 = 'o' + self.key2bz(upNodes[0])
				Input2 = 'o' + self.key2bz(upNodes[1])
				Values = self.getValue(key)
				outputsize = 0
				inputsize = 0
				for v in Values:
					if 'IO' in v['text']:
						tmp = v['text'].split(':', 2)
						inputsize = int(tmp[1])
						outputsize = int(tmp[2])
				s = 'ConcateGate(T.Gate,Input1="' + Input1 + '",Input2="' + Input2 + '",o="' + o + '")'
				l = Link(upNodes[0], key, upNodes[1], 'concateGate' + bz,s,inputsize=inputsize,outputsize=outputsize)
				self.Gates.append(l)

			elif text.lower() == 'mul':
				upNodes = self.getInputNodeKey(key)
				Input1 = 'o' + self.key2bz(upNodes[0])
				Input2 = 'o' + self.key2bz(upNodes[1])
				s = 'MulGate(T.Gate, Input1="' + Input1 + '",Input2="' + Input2 + '",o="' + o + '")'
				l = Link(upNodes[0], key, upNodes[1], 'mulGate' + bz,s)
				self.Gates.append(l)
			elif text == 'Add':
				upNodes = self.getInputNodeKey(key)
				Input1 = 'o' + self.key2bz(upNodes[0])
				Input2 = 'o' + self.key2bz(upNodes[1])
				s = 'AddGate(T.Gate, Input1="' + Input1 + '",Input2="' + Input2 + '",o="' + o + '")'
				l = Link(upNodes[0], key, upNodes[1], 'addGate' + bz,s)
				self.Gates.append(l)
			elif text.lower() == 'copy':
				upNodes = self.getInputNodeKey(key)
				Input = 'o' + self.key2bz(upNodes[0])
				s = 'CopyGate(T.Gate, Input="' + Input + '",o="' + o + '")'
				l = Link(upNodes[0], key, 0, 'copyGate' + bz,s)
				self.Gates.append(l)
			elif text.lower() == 'inout':
				upNodes = self.getInputNodeKey(key)
				Input = 'o' + self.key2bz(upNodes[0])
				Values = self.getValue(key)
				activeFunc = None
				for v in Values:
					if 'Activator' in v['text']:
						activeFunc = v['text'] + '()'
				if activeFunc == None and hasattr(self.param, 'ActiveFunc'):
					activeFunc = self.param.ActiveFunc
				s = 'InOutGate(T.Gate, Input="' + Input + '",o="' + o + '",activeFunc='+ activeFunc +')'
				l = Link(upNodes[0], key, 0, 'inoutGate' + bz,s)
				self.Gates.append(l)
			elif text.lower() == 'cnn':
				w = 'w' + bz
				bias = 'b' + bz
				upNodes = self.getInputNodeKey(key)
				Input = 'o' + self.key2bz(upNodes[0])

				Values = self.getValue(key)
				F = 2
				S = 1
				P = 0
				N = 1
				activeFunc = None
				outputsize = 0
				inputsize = 0
				channel_in = '1'
				channel_out = '1'
				for v in Values:
					if 'Activator' in v['text']:
						activeFunc = v['text'] + '()'
					elif 'IO' in v['text']:
						tmp = v['text'].split(':', 2)
						inputsize = int(tmp[1])
						outputsize = int(tmp[2])
					elif 'FSPN' == (v['text'])[:4]:
						tmp = v['text'].split(':', 4)
						F = (tmp[1])
						S = (tmp[2])
						P = (tmp[3])
						N = (tmp[4])
					elif 'CHANNEL' == (v['text'])[:7]:
						tmp = v['text'].split(':', 2)
						channel_in = (tmp[1])
						channel_out = tmp[2]
				if activeFunc == None and hasattr(self.param, 'ActiveFunc'):
					activeFunc = self.param.ActiveFunc
				if inputsize == 0 and hasattr(self.param,'w_input'):
					inputsize = self.param.w_input
				if outputsize == 0 and hasattr(self.param,'w_output'):
					outputsize = self.param.w_output
				s = 'CNNGate(T.Gate,Input="' + Input + '",W="' + w + '",bias="' + bias + '",o="' + o + '",activeFunc=' + activeFunc + ',fliters='+F+',step='+ S +',padding='+P+\
					',channel_in='+ channel_in +',channel_out=' + channel_out + ')'
				l = Link(upNodes[0], key, 0, 'cnnGate' + bz, s,inputsize=N, outputsize=F)
				l.channel_in = channel_in
				l.channel_out = channel_out
				self.Gates.append(l)
				self.WS.append(w)
				self.BiasS.append(bias)

			elif text.lower() == 'pool':
				upNodes = self.getInputNodeKey(key)
				Input = 'o' + self.key2bz(upNodes[0])
				Values = self.getValue(key)
				PoolType = None
				F = 2
				S = 1
				P = 0
				N = 1
				for v in Values:
					if 'PoolType:' == (v['text'])[:9]:
						PoolType = (v['text'])[9:]
					elif 'FSPN' == (v['text'])[:4]:
						tmp = v['text'].split(':', 4)
						F = (tmp[1])
						S = (tmp[2])
						P = (tmp[3])
						N = (tmp[4])
				#if PoolType == None and hasattr(self.param, 'PoolType'):
				#	activeFunc = self.param.ActiveFunc
				s = 'PoolGate(T.Gate, Input="' + Input + '",o="' + o + '",type="'+ PoolType +'",fliters='+F+',step='+ S +',padding='+P+',F_num='+ N +')'
				l = Link(upNodes[0], key, 0, 'poolGate' + bz, s)
				self.Gates.append(l)

			elif text.lower() == 'flatten':
				upNodes = self.getInputNodeKey(key)
				Input = 'o' + self.key2bz(upNodes[0])
				s = 'FlattenGate(T.Gate, Input="' + Input + '",o="' + o + '")'
				l = Link(upNodes[0], key, 0, 'flattenGate' + bz, s)
				self.Gates.append(l)
			elif node['figure'] != 'Value' and node['figure'] != 'Database' and node['figure'] != 'Output':
				upNodes = self.getInputNodeKey(key)
				Input = 'o' + self.key2bz(upNodes[0])
				s = 'CopyGate(T.Gate, Input="' + Input + '",o="' + o + '")'
				l = Link(upNodes[0], key, 0, 'copyGate' + bz,s)
				self.Gates.append(l)
			elif node['figure'] == 'Output':
					upNodes = self.getInputNodeKey(key)
					Input = 'o' + self.key2bz(upNodes[0])
					s = 'CopyGate(T.Gate, Input="' + Input + '",o="' + o + '")'
					l = Link(upNodes[0], key, 0, 'copyGate' + bz, s)
					self.Gates.append(l)
					self.Output.append(node)
			elif node['figure'] == 'Database':
				self.DB.append(node)




if __name__ == '__main__':
	gates = []
	model = JsonModel(gates).run()
	for g in gates:
		print(g.Value)