import pickle
import numpy as np
import os

def load_cifar_batch(filename):
	with open(filename, 'rb') as f:
		datadict = pickle.load(f, encoding='bytes')
		x = datadict[b'data']
		y = datadict[b'labels']
		x = x.reshape(x.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
		y = np.array(y)
	return x, y


def load_cifar10(root):
	xtrain = []
	ytrain = []
	for b in range(1, 6):
		f = os.path.join(root, 'data_batch_%d' % b)
		x, y = load_cifar_batch(f)
		xtrain.append(x)
		ytrain.append(y)
	X_train = np.concatenate(xtrain)
	y_train = np.concatenate(ytrain)
	X_test, y_test = load_cifar_batch(os.path.join(root, 'test_batch'))

	return X_train, y_train, X_test, y_test