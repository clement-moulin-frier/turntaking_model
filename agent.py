from copy import deepcopy
from numpy.linalg import norm
from numpy.random import rand, randn
from explauto.utils.observer import Observable
from sklearn.mixture import sample_gaussian
from numpy import array, ones, hstack, tanh, argmax


class Agent(Observable):
    def __init__(self, id, mean, covar, gmm):
        Observable.__init__(self)
        self.id = id
        self.n_ag = gmm.n_components
        self.mean, self.covar = mean, covar
        self.gmm = gmm
        self.presence = (1.) * ones(self.n_ag)
        self.lr = 0.05  # learning rate
        self.weights = self.lr * randn(2 * self.n_ag + 1) # array([0.0] * (2 * self.n_ag + 1))
        self.last_time_identified = [0] * self.n_ag
        self.t = 0

    def produce(self):
        self.activation = self.weights.dot(hstack((1., self.presence, 1. - self.presence)).T)
        self.activation = (1. + tanh(self.activation * 8.)) / 2.  # in [0, 1]
        if self.activation > rand():
            self.m = sample_gaussian(self.mean, self.covar)
        else:
            self.m = None
        self.emit("motor", (self.id, self.m))
        self.emit("activation", (self.id, self.activation))
        return self.m

    def perceive(self, s):
        bool_ags = [ss is not None for ss in s]
        voc_ag_id = [i for i, elem in enumerate(bool_ags) if elem]
        n_voc_ag = len(voc_ag_id)

        prev_presence = deepcopy(self.presence)
        self.presence_gradient = 0
        if n_voc_ag == 1:
            probas = self.gmm.predict_proba(s[voc_ag_id[0]])
            ag_id = argmax(probas)
            non_voc = []
            for i in range(self.n_ag):
                if i == ag_id:
                    self.presence_gradient += 1.1 - self.presence[i]
                    self.presence[i] += 1.
                else:
                    self.presence[i] -= 0.1
                    non_voc.append(i)
            self.presence_gradient += min(self.presence[non_voc]) - 1
        else:
            self.presence_gradient = min(self.presence) - 1.
            self.presence -= 0.1

        self.presence[self.presence > 1] = 1
        self.presence[self.presence < 0] = 0

        self.emit("presence", (self.id, deepcopy(self.presence)))
        #self.presence_gradient = float(self.presence > 0.6  - prev_presence > 0.6)
        #print self.presence_gradient

       #  self.presence_gradient = sum(self.presence - prev_presence)
        # if all(self.presence > 0.6) > 0:
        #     self.presence_gradient = 1
        # elif sum(self.presence > 0.6) - sum(prev_presence > 0.6) > 0:
        #     self.presence_gradient = 1
        # else:
        #     self.presence_gradient = -1

        self.emit("gradient", (self.id, self.presence_gradient))
        self.weights += self.lr * (self.presence_gradient) * \
            hstack((1., self.presence, 1. - prev_presence)) * (-1 if self.m is None else 1)
        self.emit("weights", (self.id, self.weights))
        self.t += 1
