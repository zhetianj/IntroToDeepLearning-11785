# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)
        """

        # if eval:
        #    # ???
        if eval:
            return self.gamma*(x-self.running_mean)/np.sqrt(self.running_var+self.eps) + self.beta

        self.x = x
        # self.mean = # ???
        self.mean = np.mean(x, axis = 0, keepdims = True)
        # self.var = # ???
        self.var = np.var(x, axis = 0, keepdims = True)
        # self.norm = # ???
        self.norm = (x - self.mean)/np.sqrt(self.var + self.eps)
        # self.out = # ???
        self.out = self.norm*self.gamma + self.beta

        # Update running batch statistics
        # self.running_mean = # ???
        self.running_mean = self.alpha*self.running_mean + (1 - self.alpha)*self.mean
        # self.running_var = # ???
        self.running_var = self.alpha*self.running_var + (1 - self.alpha)*self.var

        return self.out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """

        

        self.dgamma = np.sum(self.norm*delta, axis = 0, keepdims = True)

        self.dbeta = np.sum(delta, axis = 0, keepdims = True)

        dnorm = delta*self.gamma

        dvar = -(1/2)*np.sum(dnorm*(self.x-self.mean)*(self.var+self.eps)**(-3/2), axis = 0, keepdims = True)

        dmean = - np.sum(dnorm*(self.var+self.eps)**(-1/2), axis = 0, keepdims = True) - 1/2*np.sum(dnorm*(self.x-self.mean)*(-2*np.sum(self.x-self.mean, axis = 0, keepdims = True)/self.x.shape[0])*(self.var+self.eps)**(-3/2), axis = 0, keepdims = True)

        self.dx = dnorm*(self.var+self.eps)**(-1/2) + dvar*(2*(self.x-self.mean)/self.x.shape[0]) + dmean/self.x.shape[0]


        return self.dx
