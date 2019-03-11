import dill
import os

class Save(object):
	def __init__(self,filename,_class):
		self.filename = filename
		self._class = _class
	def _Exists(self):
		return os.path.exists(self.filename)
	def store(self):
		dill.dump_session(self.filename,main=self._class)
	def load(self):
		if self._Exists():
			dill.load_session(self.filename,main=self._class)
		else:
			print('pkl filename is not exists!')