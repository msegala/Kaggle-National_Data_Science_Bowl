import cPickle as pickle
from datetime import datetime
import os
import sys
import gzip
from matplotlib import pyplot
import numpy as np
from lasagne import layers
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import theano
from lasagne import nonlinearities
from autoencoder import  AutoEncoder

Conv2DLayer = layers.Conv2DLayer
MaxPool2DLayer = layers.MaxPool2DLayer

sys.setrecursionlimit(10000)  # for pickle...
np.random.seed(42)

FTRAIN = '/Users/msegala/Documents/Personal/Kaggle-National_Data_Science_Bowl/Data/mnist.pkl.gz'
FTEST = '/Users/msegala/Documents/Personal/Kaggle-National_Data_Science_Bowl/Data/mnist.pkl.gz'

def float32(k):
    return np.cast['float32'](k)

def _load_data(filename):
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def load(test=False, cols=None):

    fname = FTEST if test else FTRAIN
    print "Running for: ",fname

    data = _load_data(fname)

    if not test:

        X, y = data[2]

        X = X.astype(np.float32)
        y = y.astype(np.int32)

        print X.shape
        print y.shape

        X, y = shuffle(X, y, random_state=42)  # shuffle train data
    else:

        X, y = data
        X = X.astype(np.float32)
        y = y.astype(np.int32)

        print X.shape
        print y.shape

    return X, y


def load2d(test=False, cols=None):
    X, y = load(test=test, cols=cols)
    X = X.reshape(-1, 1, 28, 28)
    return X, y


#Auto = layers.AutoEncoder

net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('auto', AutoEncoder),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 28, 28),

    auto_num_units = 28*28, 
    auto_n_hidden = 28*28,

    output_num_units=28*28, 

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),
    output_nonlinearity=nonlinearities.softmax,

    regression=True,
    max_epochs=3,
    verbose=1,
    )


def fit():
    X, y = load2d()
    net.fit(X, y)
    with open('net.pickle', 'wb') as f:
        pickle.dump(net, f, -1)


if __name__ == '__main__':
    fit()
