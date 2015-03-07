from copy import deepcopy
from numpy.linalg import norm
from numpy.random import rand, randn
from explauto.utils.observer import Observable
from sklearn.mixture import sample_gaussian
from numpy import array, ones, hstack, tanh, argmax, product, zeros, sign


class Agent(Observable):
    def __init__(self, id, mean, covar, gmm):
        Observable.__init__(self)
        self.id = id
        self.n_ag = gmm.n_components
        self.mean, self.covar = mean, covar
        self.gmm = gmm
        self.presence = hstack((1., (1./self.n_ag) * ones(self.n_ag)))
        self.lr = 0.05  # learning rate
        self.weights = self.lr * randn(self.n_ag + 1) # array([0.0] * (2 * self.n_ag + 1))
        self.value_weights = self.lr * randn(self.n_ag + 1)
        self.discount = 0.9
        self.last_time_identified = [0] * self.n_ag
        self.t = 0
        self.lr_alpha = 0. #99  # decaying lr(t) = t ** (-alpha) according to Damas' IML algorithm
                              # Check if we have sum_t lr(t) == inf and sum_t lr(t)**2 < inf (Eq 21, 22 in Grondman et al.IEEE SMC)
        # self.topics = ["motor", "presence", "td_error", "activation", "weights"]

    def produce(self):
        self.activation = self.weights.dot(self.presence.T)
        self.activation = (1. + tanh(self.activation * 1.)) / 2.  # in [0, 1]
        if self.activation > rand():
            self.m = sample_gaussian(self.mean, self.covar)
        else:
            self.m = None
        self.emit("motor", (self.id, self.m))
        self.emit("activation", (self.id, self.activation))
        return self.m

    def perceive(self, s):
        self.lr = 0.01 * (self.t + 1) ** (- self.lr_alpha)

        bool_ags = [ss is not None for ss in s]
        voc_ag_id = [i for i, elem in enumerate(bool_ags) if elem]
        n_voc_ag = len(voc_ag_id)

        prev_presence = deepcopy(self.presence)
        if n_voc_ag == 1:
            probas = self.gmm.predict_proba(s[voc_ag_id[0]])
            ag_id = argmax(probas)
            non_voc = []
            for i in range(self.n_ag):
                if i == ag_id:
                    self.presence[i + 1] += 1.
                else:
                    self.presence[i + 1] -= 0.1
                    non_voc.append(i)
        else:
            self.presence[1:] -= 0.1

        self.presence[self.presence > 1] = 1
        self.presence[self.presence < 0] = 0


        reward = product(self.presence[1:])

        value_prev_state = self.value_weights.dot(prev_presence.T)
        value_current_state = self.value_weights.dot(self.presence.T)

        td_error = reward + self.discount * value_current_state - value_prev_state

        self.value_weights += self.lr * td_error * prev_presence

        self.weights += self.lr * td_error * prev_presence * (-1 if self.m is None else 1) #* self.activation * (1. - self.activation) #  (-1 if self.m is None else 1)

        #self.presence_gradient = float(self.presence > 0.6  - prev_presence > 0.6)
        #print self.presence_gradient

       #  self.presence_gradient = sum(self.presence - prev_presence)
        # if all(self.presence > 0.6) > 0:
        #     self.presence_gradient = 1
        # elif sum(self.presence > 0.6) - sum(prev_presence > 0.6) > 0:
        #     self.presence_gradient = 1
        # else:
        #     self.presence_gradient = -1

        
        #self.weights += self.lr * (self.presence_gradient) * \
        #    hstack((1., self.presence, 1. - prev_presence)) * (-1 if self.m is None else 1)
        
        self.t += 1

        self.emit("presence", (self.id, deepcopy(self.presence)))
        self.emit("td_error", (self.id, td_error))
        self.emit("weights", (self.id, self.weights))

