import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		#for idx_cls in node.num_cls:
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			branches = np.array(branches).astype(int)
			Total_Branch_TrainEx = np.sum(branches, axis=0)
			def kk(x):
				x[x<=0]=1
				return -1*x*np.log2(x)
			return np.sum(np.sum(np.apply_along_axis(kk,axis=1,arr=(branches / (Total_Branch_TrainEx.reshape(1,Total_Branch_TrainEx.size)))), axis=0) * (Total_Branch_TrainEx / np.sum(Total_Branch_TrainEx)))

		MinimumEntropy = float('inf')
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################

			ff=np.array(self.features)
			feature_values = ff[:, idx_dim]
			if None in feature_values:continue
			BV = np.unique(feature_values)
			BR = np.zeros((self.num_cls, len(BV)))
			BVL=BV.tolist()
			def vectrize_Z(x):
				LAbels=np.array(self.labels)[np.where(feature_values == x)]
				pos=BVL.index(x)
				for LAbel in LAbels: BR[LAbel, pos] = BR[LAbel, pos]+1

			np.vectorize(vectrize_Z)(BV)
			branch_Entropy = conditional_entropy(BR)
			if branch_Entropy < MinimumEntropy:
				self.feature_uniq_split = BVL
				MinimumEntropy = branch_Entropy
				self.dim_split = idx_dim



		############################################################
		# TODO: split the node, add child nodes
		############################################################

		Feature_Remain=np.array(self.features)
		Split_Feature=Feature_Remain[:,self.dim_split]
		Feature_Remain=Feature_Remain.astype(object)
		Feature_Remain[:,self.dim_split]=None
		#Feature_Remain = np.delete(Feature_Remain, self.dim_split, axis=1)
		iL = len(self.feature_uniq_split)
		i=0
		while i<iL:
			Pos=np.where(Split_Feature==self.feature_uniq_split[i])
			Node_Child=TreeNode(Feature_Remain[Pos].tolist(),np.array(self.labels)[Pos].tolist(),self.num_cls)
			if all(Feature_Remain_Value is None for Feature_Remain_Value in Feature_Remain[Pos].tolist()[0]): Node_Child.splittable = False
			if Feature_Remain[Pos].size == 0:Node_Child.splittable = False
			self.children.append(Node_Child)
			i+=1

		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			#feature = feature[:self.dim_split] + feature[self.dim_split + 1:]
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



