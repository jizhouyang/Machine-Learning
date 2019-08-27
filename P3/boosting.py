import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		#'''
		FinalH=np.zeros(len(features))
		t=0
		k1=len(self.clfs_picked)
		k2=len(self.betas)
		while(t<k1 and t<k1):
			ht=np.array(self.clfs_picked[t].predict(features))
			FinalH=FinalH+self.betas[t]*ht
			t+=1;
		FinalH[FinalH>0]=1
		FinalH[FinalH<=0]=-1
		FinalH=FinalH.astype(int).tolist()
		return FinalH
		#'''
		########################################################

		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"

		weight=(1/len(labels))*np.ones(len(labels))


		t=0

		def ff(x):
			h = x.predict(features)
			return np.sum(weight * (np.array(labels) != np.array(h)))
		while t<self.T:

			BE = float("inf")


			kk=list(self.clfs)
			classfier=np.array(kk)
			ff1=np.vectorize(ff)
			classfier=ff1(classfier)
			BE=np.min(classfier)
			ht=kk[np.argmin(classfier)]
			self.clfs_picked.append(ht)
			MinH=ht.predict(features)



			logfactor=(1 - BE) / BE
			B=0.5 * np.log(logfactor)
			self.betas.append(B)




			beta1=np.exp(-B)
			beta2=np.exp(B)
			temp=weight*(beta1*(np.array(np.array(labels)==np.array(MinH)).astype(int)))
			weight=weight*(beta2*(np.array(np.array(labels)!=np.array(MinH)).astype(int)))+temp
			#print(weight)


			weight = weight/np.sum(weight)
			t+=1

		############################################################
		
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	