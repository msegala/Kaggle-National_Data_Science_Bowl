"""
To use this script:
	> python run_analysis.py fit 
	This will save the output to net-specialists.pickle

	 -OR-

	> python run_analysis.py fit name.pickle
	This will save the output to name.pickle
"""

from helpers import *
from reshape import *

net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('noise1', layers.GaussianNoiseLayer),
        ('conv1', Conv2DLayer),
        ('pool1', MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', Conv2DLayer),
        ('pool2', MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', Conv2DLayer),
        ('pool3', MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
        ('conv4', Conv2DLayer),
        ('pool4', MaxPool2DLayer),
        ('dropout4', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),
        ('dropout5', layers.DropoutLayer),
        ('maxout5', Maxout),
        ('hidden6', layers.DenseLayer),
        ('dropout6', layers.DropoutLayer),
        ('maxout6', Maxout),
        ('output', layers.DenseLayer),
        ],

    input_shape=(None, 1, PIXELS, PIXELS),
    conv1_num_filters=208, conv1_filter_size=(6, 6), pool1_ds=(2, 2),
    dropout1_p=0.2,
    conv2_num_filters=384, conv2_filter_size=(4, 4), pool2_ds=(2, 2),
    dropout2_p=0.2,
    conv3_num_filters=512, conv3_filter_size=(3, 3), pool3_ds=(2, 2),
    dropout3_p=0.2,
    conv4_num_filters=896, conv4_filter_size=(3, 3), pool4_ds=(2, 2),
    dropout4_p=0.2,

    #conv1_untie_biases = True,conv2_untie_biases = True,conv3_untie_biases = True,conv4_untie_biases = True,
    
    hidden5_num_units=8192,
    dropout5_p=0.5,
    maxout5_ds=2,
    hidden5_nonlinearity=nonlinearities.sigmoid,
    
    hidden6_num_units=8192,
    dropout6_p=0.5,
    maxout6_ds=2,
    hidden6_nonlinearity=nonlinearities.sigmoid,
    
    output_num_units=PIXELS*PIXELS, 
    output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.01)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    #batch_iterator_train=FlipBatchIterator(batch_size=128),
    #batch_iterator_train=DataAugmentationBatchIterator(batch_size=128),
    #batch_iterator_test=BatchIterator(batch_size=128),
    #on_epoch_finished=[
        #AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        #AdjustVariable('update_learning_rate', start=0.03, stop=0.001),
        #AdjustVariable('update_momentum', start=0.9, stop=0.999),
    #    EarlyStopping(patience=30),
    #    ],
    max_epochs=18,
    #max_epochs=100,
    verbose=1,
    eval_size=0.0
    )



def fit(output='net-specialists_autoencoder.pickle', fname_pretrain=None):

    print "Saving Output to:", output
    print "Running for Pixels = ",PIXELS

    X_train, y_train = load2d()
    print "Training shape = ",X_train.shape

    X_test, y_test = load2d(test=True)
    print "Testing shape = ",X_test.shape

    X = np.vstack((X_train,X_test))
    print "Final shape = ",X.shape
    
    net.fit(X, X.reshape(-1,3600))
    #net.save_weights_to("autoencoder_weights_test_and_train.pickle")
        
    with open(output, 'wb') as f:
        pickle.dump(net, f, -1)

    


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        func = globals()[sys.argv[1]]
        func(*sys.argv[2:])

