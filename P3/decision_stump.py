import numpy as np
from typing import List
from classifier import Classifier

class DecisionStump(Classifier):
	def __init__(self, s:int, b:float, d:int):
		self.clf_name = "Decision_stump"
		self.s = s
		self.b = b
		self.d = d

	def train(self, features: List[List[float]], labels: List[int]):
		pass
		
	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		##################################################
		# TODO: implement "predict"

		Trans_features=np.array(features)[:,self.d]
		if self.s > self.b:
			Trans_features[Trans_features > self.b] = self.s
			Trans_features[Trans_features<=self.b]=-self.s
			Trans_features = Trans_features.tolist()
			return Trans_features
		else:
			Trans_features=Trans_features.tolist()
			fh=[]
			for f in Trans_features:
				if f>self.b:fh.append(self.s)
				else:fh.append(-self.s)
			return fh

		##################################################
		