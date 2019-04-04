from data_utlis import load_cifar10
from k_nearest_neighbor import KNearestNeighbor
import numpy as np


cifar_10_dir = './cifar-10-batches-py'
x_train, y_train, x_test, y_test = load_cifar10(cifar_10_dir)
print('train_data_shape:', x_train.shape)
print('train_labels_shape:', y_train.shape)
print('test_data_shape:', x_test.shape)
print('test_labels_shape:', y_test.shape)

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
num_train = x_train.shape[0]
num_test = x_test.shape[0]


# num_train = 5000
# mask = range(num_train)
# x_train = x_train[mask]
# y_train = y_train[mask]
# num_test = 500
# mask = range(num_test)
# x_test = x_test[mask]
# y_test = y_test[mask]

classifier = KNearestNeighbor()
classifier.train(x_train, y_train)
dicts = classifier.compute_distance(x_test)
y_test_pred = classifier.predict_labels(dicts, k=10)

num_correct = np.sum(y_test_pred == y_test)
accuracy = num_correct / num_test
print('got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))






