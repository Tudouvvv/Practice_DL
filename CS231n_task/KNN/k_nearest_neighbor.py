import numpy as np

class KNearestNeighbor():
	def __init__(self):
		pass

	def train(self, x, y):
		self.x_train = x
		self.y_train = y


	def compute_distance(self, x):
		num_test = x.shape[0]
		num_train = self.x_train.shape[0]
		dicts = np.zeros((num_test, num_train))
		for i in range(num_test):
			for j in range(num_train):
				dicts[i, j] = np.sqrt(np.sum((x[i,:] - self.x_train[j, :])**2))
		return dicts


	# def compute_distance_no_loops(self, x):
	# 	num_test = x.shape[0]
	# 	num_train = self.x_train.shape[0]
	# 	dicts = np.zeros((num_test, num_train))
	# 	for i in range(num_test):
	# 		for j in range(num_train):
	# 			dicts[i] = np.sum(np.abs(self.x_train[j, :] - x[i, :]))
	# 	return dicts
	
	def predict_labels(self, dists, k=1):
		num_test = dists.shape[0]
		y_pred = np.zeros(num_test)
		for i in range(num_test):
			closest_y = []
			y_indicies = np.argsort(dists[i,:], axis=0)
			closest_y = self.y_train[y_indicies[ : k]]
			y_pred[i] = np.argmax(np.bincount(closest_y))
		return y_pred