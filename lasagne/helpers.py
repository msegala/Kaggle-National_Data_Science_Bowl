"""
This script is used as helpers.
"""

import gzip
import itertools
import urllib
import pickle
from datetime import datetime
import os
import sys
import copy
from matplotlib import pyplot
import numpy as np
from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import theano
from lasagne.updates import nesterov_momentum
from collections import OrderedDict
from sklearn.base import clone

from skimage import transform
import skimage

PIXELS = 60
FTRAIN = '../Data/Train_Data_60x60_pixels.pkl.gz'
FTEST = '../Data/Test_Data_60x60_pixels.pkl.gz'
USE_GPU = False

sys.setrecursionlimit(150000)  # for pickle...
np.random.seed(42)


if USE_GPU:
    Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
    MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer
else:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer

Maxout = layers.pool.FeaturePoolLayer
#Shape = layers.shape.reshape


SPECIALIST_SETTINGS = [dict(columns=('ALL',),),]

class DataAugmentationBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        Xb, yb = super(DataAugmentationBatchIterator, self).transform(Xb, yb)

        augmentation_params = {
            'zoom_range': (1.0, 1.1),
            'rotation_range': (0, 360),
            'shear_range': (0, 20),
            'translation_range': (-10, 10),
        }

        IMAGE_WIDTH = PIXELS
        IMAGE_HEIGHT = PIXELS

        def fast_warp(img, tf, output_shape=(PIXELS,PIXELS), mode='nearest'):
            """
            This wrapper function is about five times faster than skimage.transform.warp, for our use case.
            """
            #m = tf._matrix
            m = tf.params
            img_wf = np.empty((output_shape[0], output_shape[1]), dtype='float32')
            #for k in xrange(1):
            #    img_wf[..., k] = skimage.transform._warps_cy._warp_fast(img[..., k], m, output_shape=output_shape, mode=mode)
            img_wf = skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode)
            return img_wf

        def random_perturbation_transform(zoom_range, rotation_range, shear_range, translation_range, do_flip=True):
            # random shift [-10, 10] - shift no longer needs to be integer!
            shift_x = np.random.uniform(*translation_range)
            shift_y = np.random.uniform(*translation_range)
            translation = (shift_x, shift_y)

            # random rotation [0, 360]
            rotation = np.random.uniform(*rotation_range) # there is no post-augmentation, so full rotations here!

            # random shear [0, 20]
            shear = np.random.uniform(*shear_range)

            # random zoom [0.9, 1.1]
            # zoom = np.random.uniform(*zoom_range)
            log_zoom_range = [np.log(z) for z in zoom_range]
            zoom = np.exp(np.random.uniform(*log_zoom_range)) # for a zoom factor this sampling approach makes more sense.
            # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.


            translation = (0,0)
            rotation = 0.0
            shear = 0.0
            zoom = 1.0

            rotate =  np.random.randint(6)
            if rotate == 0:
                rotation = 0.0
            elif rotate == 1:
                rotation = 45.0
            elif rotate == 2:
                rotation = 90.0
            elif rotate == 3:
                rotation = 135.0
            elif rotate == 4:
                rotation = 180.0
            else:
                rotation = 270.0


            ## flip
            if do_flip and (np.random.randint(2) > 0): # flip half of the time
                shear += 180
                rotation += 180
                # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
                # So after that we rotate it another 180 degrees to get just the flip.            

            '''
            print "translation = ", translation
            print "rotation = ", rotation
            print "shear = ",shear
            print "zoom = ",zoom
            print ""
            '''

            return build_augmentation_transform(zoom, rotation, shear, translation)


        center_shift = np.array((IMAGE_HEIGHT, IMAGE_WIDTH)) / 2. - 0.5
        tform_center = transform.SimilarityTransform(translation=-center_shift)
        tform_uncenter = transform.SimilarityTransform(translation=center_shift)

        def build_augmentation_transform(zoom=1.0, rotation=0, shear=0, translation=(0, 0)):
            tform_augment = transform.AffineTransform(scale=(1/zoom, 1/zoom), 
                                                      rotation=np.deg2rad(rotation), 
                                                      shear=np.deg2rad(shear), 
                                                      translation=translation)
            tform = tform_center + tform_augment + tform_uncenter # shift to center, augment, shift back (for the rotation/shearing)
            return tform

        tform_augment = random_perturbation_transform(**augmentation_params)
        tform_identity = skimage.transform.AffineTransform()
        tform_ds = skimage.transform.AffineTransform()
        
        for i in range(Xb.shape[0]):
            new = fast_warp(Xb[i][0], tform_ds + tform_augment + tform_identity, output_shape=(PIXELS,PIXELS), mode='nearest').astype('float32')
            Xb[i,:] = new
            
        return Xb, yb



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

        X_train, y_train = data[0]
        X_valid, y_valid = data[1]
        X_test, y_test = data[2]

        print X_train.shape
        print X_valid.shape
        print X_test.shape
        print y_train.shape
        print y_valid.shape
        print y_test.shape

        X = np.hstack(([X_train],[X_valid],[X_test]))
        y = np.hstack(([y_train],[y_valid],[y_test]))
        print X[0].shape
        print y[0].shape
        
        X = X[0].astype(np.float32)
        y = y[0].astype(np.int32)
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
    X = X.reshape(-1, 1, PIXELS, PIXELS)
    return X, y



class FlipBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]
        return Xb, yb



class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']

        ls_tmp = np.linspace(0.03, 0.0001, 50)
        ls_tmp_2 = np.linspace(0.0001, 0.000001, 50)
        ls_tmp_3 = np.linspace(0.000001, 0.0000001, 50)

        if epoch <= 50:
            new_value = np.cast['float32'](ls_tmp[epoch - 1])
        elif epoch > 50 and epoch <= 100:
            #new_value = np.cast['float32'](0.00005)
            new_value = np.cast['float32'](ls_tmp_2[epoch - 1 - 50])
        else:
            #new_value = np.cast['float32'](0.00001)
            new_value = np.cast['float32'](ls_tmp_3[epoch - 1 - 100])

        #new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()


def rebin( a, newshape ):
    from numpy import mgrid
    assert len(a.shape) == len(newshape)

    slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]


def plot_learning_curves(fname_specialists='net-specialists.pickle'):
    with open(fname_specialists, 'r') as f:
        models = pickle.load(f)

    fig = pyplot.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_color_cycle(
        ['c', 'c', 'm', 'm', 'y', 'y', 'k', 'k', 'g', 'g', 'b', 'b'])

    valid_losses = []
    train_losses = []

    for model_number, (cg, model) in enumerate(models.items(), 1):

        valid_loss = np.array([i['valid_loss'] for i in model.train_history_])
        train_loss = np.array([i['train_loss'] for i in model.train_history_])
        valid_loss = np.sqrt(valid_loss) 
        train_loss = np.sqrt(train_loss) 

        valid_loss = rebin(valid_loss, (100,))
        train_loss = rebin(train_loss, (100,))

        valid_losses.append(valid_loss)
        train_losses.append(train_loss)
        ax.plot(valid_loss,
                label='{} ({})'.format(cg[0], len(cg)), linewidth=3)
        ax.plot(train_loss,
                linestyle='--', linewidth=3, alpha=0.6, label='Train')
        ax.set_xticks([])

    weights = np.array([m.output_num_units for m in models.values()],
                       dtype=float)
    weights /= weights.sum()
    mean_valid_loss = (
        np.vstack(valid_losses) * weights.reshape(-1, 1)).sum(axis=0)
    ax.plot(mean_valid_loss, color='r', label='mean Valid', linewidth=4, alpha=0.8)

    ax.legend()
    #ax.set_ylim((0.0, 1.0))
    ax.grid()
    pyplot.ylabel("NLL")
    pyplot.xlabel("Batch")
    pyplot.show()
    pyplot.savefig('foo.png')

