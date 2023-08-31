# import lib
import torch
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import scipy.io
from sklearn.utils import shuffle
import urllib.request as urllib2
import sys
import math

# class help to make dataloader
class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, attributes, labels):
        'Initialization'
        self.labels = labels
        self.attributes = attributes

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        X = self.attributes[index]
        y = self.labels[index]
        return X, y

# function for loading data
def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # shuffle the data
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    # Normalize data
    x_train = x_train / 255.
    x_test = x_test / 255.
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # reshape the data
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

    # turn the label to one-hot
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # only use part of the data for saving time
    x_valid = x_train[5400:6000]
    y_valid = y_train[5400:6000]
    x_train = x_train[:5400]
    y_train = y_train[:5400]
    x_test = x_test[:1000]
    y_test = y_test[:1000]

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def load_madelon():
    # get data via url
    train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
    train_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
    test_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data'
    test_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels'

    train_x = np.loadtxt(urllib2.urlopen(train_data_url)).astype('float32')
    train_y = np.loadtxt(urllib2.urlopen(train_resp_url)).astype('int')
    test_x = np.loadtxt(urllib2.urlopen(test_data_url)).astype('float32')
    test_y = np.loadtxt(urllib2.urlopen(test_resp_url)).astype('int')

    # normalize data
    xTrainMean = np.mean(train_x, axis=0)
    xTrainStd = np.std(train_x, axis=0)
    train_x = (train_x - xTrainMean) / xTrainStd
    xTestMean = np.mean(test_x, axis=0)
    xTestStd = np.std(test_x, axis=0)
    test_x = (test_x - xTestMean) / xTestStd

    # turn y's -1 label to 0
    train_y[train_y == -1] = 0
    test_y[test_y == -1] = 0

    # turn the label to one-hot
    train_y = to_categorical(train_y, 2)
    test_y = to_categorical(test_y, 2)

    # split validation set from training set by 10%
    valid_x = train_x[1800:]
    valid_y = train_y[1800:]
    train_x = train_x[:1800]
    train_y = train_y[:1800]

    return train_x, train_y, valid_x, valid_y, test_x, test_y

def load_mat(filename, num_datapoint, num_class):
    # read data from file
    data = scipy.io.loadmat(filename)
    x, y = shuffle(data['X'], data['Y'])

    # normalize data
    x = x.astype('float32')
    xMean = np.mean(x, axis=0)
    xStd = np.std(x, axis=0)
    x = (x - xMean) / xStd 

    # turn the label to one-hot
    unique_classes = np.unique(y)
    if -1 in unique_classes:
        if not 0 in unique_classes:
            y[y == -1] = 0 # make the classes count from 0
        else:
            y[y== -1] = num_class - 1
    unique_classes = np.unique(y)
    if not 0 in unique_classes:
        y[y == num_class] = 0 # make the classes count from 0
    y = to_categorical(y, num_class)

    # split 20% as test set, and split 10% from train set as validation set
    valid_idx = math.ceil(num_datapoint * 0.8)
    train_idx = math.ceil(valid_idx * 0.9)

    x_train = x[:train_idx]
    y_train = y[:train_idx]
    if num_datapoint < 200:
        x_valid = x[:valid_idx]
        y_valid = y[:valid_idx]
    else:
        x_valid = x[train_idx:valid_idx]
        y_valid = y[train_idx:valid_idx]
    x_test = x[valid_idx:]
    y_test = y[valid_idx:]

    return x_train, y_train, x_valid, y_valid, x_test, y_test

# function for selecting dataset
def get_dataset(dataset_name):
    # hand-written digit image
    if dataset_name == 'mnist':
        return load_mnist()
    elif dataset_name == 'usps':
        return load_mat('./datasets/USPS.mat', 9298, 10) 
    
    # face image data
    elif dataset_name == 'coil20':
        return load_mat('./datasets/COIL20.mat', 1440, 20)

    # text dataset
    elif dataset_name == 'basehock':
        return load_mat('./datasets/BASEHOCK.mat', 1993, 2)
    elif dataset_name == 'pcmac':
        return load_mat('./datasets/PCMAC.mat', 1943, 2)

    # biological data
    elif dataset_name == 'prostate_ge':
        return load_mat('./datasets/Prostate_GE.mat', 102, 2)
    elif dataset_name == 'tox':
        return load_mat('./datasets/TOX_171.mat', 171, 4)
    
    # speech dataset
    elif dataset_name == 'isolet':
        return load_mat('./datasets/Isolet.mat', 1560, 26)
    
    # artificial dataset
    elif dataset_name == 'madelon':
        return load_madelon()
        
    else:
        print('can not find the corresponding dataset.')
        sys.exit()

