"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys
import time

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        self.linear_layers = [Linear(i, j, weight_init_fn, bias_init_fn) for i, j in zip([input_size] + hiddens, hiddens + [output_size])]

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = [BatchNorm(hiddens[i]) for i in range(num_bn_layers)]



    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """ 
        # Complete the forward pass through your entire MLP.

        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i].forward(x)
            if i < self.num_bn_layers:
                if self.train_mode:
                    x = self.bn_layers[i].forward(x)
                else:
                    x = self.bn_layers[i].forward(x, True)
            x = self.activations[i].forward(x)

        self.output = x

        return self.output

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.

        for i in self.linear_layers:
            i.dW = np.zeros(i.dW.shape)
            i.db = np.zeros(i.db.shape)

        if self.bn:
            for i in self.bn_layers:
                i.dgamma = np.zeros(i.dgamma.shape)
                i.dbeta = np.zeros(i.dbeta.shape)

        return None


    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        #for i in range(len(self.linear_layers)):
            # Update weights and biases here
        # Do the same for batchnorm layers

        for i in range(len(self.linear_layers)):
            self.linear_layers[i].momentum_W = self.momentum*self.linear_layers[i].momentum_W - self.lr*self.linear_layers[i].dW
            self.linear_layers[i].W += self.linear_layers[i].momentum_W
            self.linear_layers[i].momentum_b = self.momentum*self.linear_layers[i].momentum_b - self.lr*self.linear_layers[i].db
            self.linear_layers[i].b += self.linear_layers[i].momentum_b
            
        if self.bn:
            for i in range(len(self.bn_layers)):
                self.bn_layers[i].gamma -= self.lr*self.bn_layers[i].dgamma
                self.bn_layers[i].beta -= self.lr*self.bn_layers[i].dbeta

        return None

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.

        sml = SoftmaxCrossEntropy()
        sml.forward(self.output, labels)
        sml_derivative = sml.derivative()

        for i in range(len(self.linear_layers)):
            sml_derivative = np.multiply(self.activations[-1-i].derivative(), sml_derivative)
            if i >= len(self.linear_layers) - self.num_bn_layers:
                sml_derivative = self.bn_layers[len(self.linear_layers) - i - 1].backward(sml_derivative)
            sml_derivative = self.linear_layers[-1-i].backward(sml_derivative)

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...
    mlp = MLP(trainx.shape[1], trainy.shape[1], [128, 64], [ReLU(), ReLU(), ReLU()], 
        	np.random.randn, lambda x: np.random.randn(1, x), SoftmaxCrossEntropy(), 0.0001, 1.0, 2)

    print(nepochs)
    for e in range(nepochs):

        # Per epoch setup ...
        
        print("Start epoch {0}...".format(e+1))
        training_loss = 0
        training_error = 0

        validation_loss = 0
        validation_error = 0

        for b in range(0, len(trainx), batch_size):

            # Train ...
            mlp.train()
            index = idxs[[i for i in range(b, b+batch_size) if i < len(trainx)]]
            train_input = trainx[index]
            output = mlp.forward(train_input)
            mlp.backward(trainy[index])
            mlp.step() 



            training_loss += mlp.total_loss(trainy[index])
            training_error += mlp.error(trainy[index])

        for b in range(0, len(valx), batch_size):

            # Val ...
            mlp.eval()
            index = [i for i in range(b, b+batch_size) if i < len(valx)]

            val_input = valx[index]
            output = mlp.forward(val_input)

            validation_loss += mlp.total_loss(valy[index])
            validation_error += mlp.error(valy[index])

        # Accumulate data...
        
        print("train loss: {0}".format(training_loss/len(trainx)))
        print("train error: {0}".format(training_error/len(trainx)))
        print("val loss: {0}".format(validation_loss/len(valx)))
        print("val error: {0}".format(validation_error/len(valx)))
        print("Done epoch {0}...".format(e+1))
        print("==============================")
        np.random.shuffle(idxs)
        mlp.zero_grads()


        training_losses[e] = training_loss/len(trainx)
        training_errors[e] = training_error/len(trainx)
        validation_losses[e] = validation_loss/len(valx)
        validation_errors[e] = validation_error/len(valx)

    return (training_losses, training_errors, validation_losses, validation_errors)

''''
        training_losses[e] = training_loss/len(range(0, len(trainx), batch_size))
        training_errors[e] = training_error/len(range(0, len(trainx), batch_size))
        validation_losses[e] = validation_loss/len(range(0, len(valx), batch_size))
        validation_errors[e] = validation_error/len(range(0, len(valx), batch_size))
'''
    # Cleanup ...

    # Return results ...
    
