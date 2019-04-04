from data_utlis import load_cifar10
from k_nearest_neighbor import KNearestNeighbor
import numpy as np
import matplotlib.pyplot as plt

cifar_10_dir = './cifar-10-batches-py'
x_train, y_train, x_test, y_test = load_cifar10(cifar_10_dir)
print('train_data_shape:', x_train.shape)
print('train_labels_shape:', y_train.shape)
print('test_data_shape:', x_test.shape)
print('test_labels_shape:', y_test.shape)

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

num_train = 5000
mask = range(num_train)
x_train = x_train[mask]
y_train = y_train[mask]
num_test = 500
mask = range(num_test)
x_test = x_test[mask]
y_test = y_test[mask]

num_folds=5
k_choices=[1,3,5,8,10,12,15,20,50,100]

y_train=y_train.reshape(-1,1)

x_train_folds=np.array_split(x_train, num_folds)
y_train_folds=np.array_split(y_train, num_folds)

x_train_folds = np.array(x_train_folds)
y_train_folds = np.array(y_train_folds)


k_to_accuracies={}

for k in k_choices:
    k_to_accuracies.setdefault(k,[])

for i in range(num_folds):
    classifier=KNearestNeighbor()
    x_val_train = np.concatenate((x_train_folds[0:i], x_train_folds[i+1:]), axis=0)
    x_val_train = x_val_train.reshape(-1, x_val_train.shape[2])
    y_val_train = np.concatenate((y_train_folds[0:i], y_train_folds[i+1:]), axis=0)
    y_val_train = y_val_train.reshape(-1, y_val_train.shape[2])

    y_val_train=y_val_train[:,0]
    classifier.train(x_val_train,y_val_train)
    for k in k_choices:
        y_val_pred=classifier.predict_labels(x_train_folds[i],k=k)
        num_correct=np.sum(y_val_pred==y_train_folds[i][:,0])
        accuracy=float(num_correct)/len(y_val_pred)
        k_to_accuracies[k]=k_to_accuracies[k]+[accuracy]

for k in k_choices:
    accuracies=k_to_accuracies[k]
    plt.scatter([k]*len(accuracies),accuracies)

accuracies_mean=np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std=np.array([np.std(v) for k ,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices,accuracies_mean,yerr=accuracies_std)
plt.title('cross-validation on k')
plt.xlabel('k')
plt.ylabel('cross-validation accuracy')
plt.show()