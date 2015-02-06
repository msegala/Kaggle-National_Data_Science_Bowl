import numpy as np
import theano
import theano.tensor as T

from .. import init
from .. import nonlinearities

from .base import Layer


__all__ = [
    "AutoEncoder",
]

class AutoEncoder(Layer):
    """ asdsa """
    def __init__(self, incoming, num_units, n_hidden, W=init.Uniform(), bhid=init.Constant(0.), bvis=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, **kwargs):
        super(AutoEncoder, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity
 
        self.num_units = num_units
        self.n_hidden = n_hidden
        self.x = incoming
 
        num_inputs = int(np.prod(self.input_shape[1:]))
 
        initial_W = numpy.asarray(
            numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + num_units)),
                    high=4 * np.sqrt(6. / (n_hidden + num_units)),
                    size=(num_units, n_hidden)
            ),
            dtype=theano.config.floatX
        )
            
        self.W = self.create_param(initial_W, (num_inputs, n_hidden), name="W")
        self.bvis = self.create_param(bvis, (num_units,), name="bvis") if bvis is not None else None
        self.bhid = self.create_param(bhid, (n_hidden,), name="bhid") if bhid is not None else None
 
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        
        
    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input
 
    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)
 
    def get_reconstructed_input(self, hidden):
        """ Computes the reconstructed input given the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
 
    def get_params(self):
        return [self.W] + self.get_bias_params()
 
    def get_bias_params(self):
        return [self.b, self.b_prime] if self.b is not None else []
        
    def get_output_shape_for(self, input_shape):
        return self.n_hidden
 
    def get_output_for(self, input, *args, **kwargs):
 
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        activation = T.mean(L)
 
        return self.nonlinearity(activation)