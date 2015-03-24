from copy import deepcopy
from collections import deque
from numpy.linalg import norm
from sklearn.mixture import GMM
from numpy.random import rand, randn
from sklearn.mixture import sample_gaussian
from explauto.utils.observer import Observable
from numpy import array, ones, hstack, tanh, argmax, product, zeros, sign, mean


class FeatureExtraction(object):
    def __init__(self):
        self.amp_hist = deque(maxlen=1)

    def features(self, signal):
        return signal

    def amplitude(self, signal):
        amp = 1.0 if any([s is not None for s in signal]) else 0.
        self.amp_hist.append(amp)
        return mean(self.amp_hist)


class AgentIndentification(object):
    def __init__(self, ag_voc_params):
        n_ag = len(ag_voc_params)
        self.gmm = GMM(n_ag, covariance_type='full')
        self.gmm.means_, self.gmm.covars_ = array([voc[1] for voc in ag_voc_params]), array([voc[2] for voc in ag_voc_params])
        self.gmm.weights_ = array([1. / n_ag] * n_ag)

    def identify(self, features):
        bool_ags = [f is not None for f in features]
        voc_ag_id = [i for i, elem in enumerate(bool_ags) if elem]
        n_voc_ag = len(voc_ag_id)
        if n_voc_ag == 1:
            probas = self.gmm.predict_proba(features[voc_ag_id[0]])
            ag_id = argmax(probas)
            return ag_id
        else:
            return None


class PresenceEstimation(object):
    def __init__(self, n_ag):
        self.n_ag = n_ag
        self.presence = (1. / n_ag) * ones(n_ag)
        self.last_time_identified = [0] * n_ag
        self.onset = 4
        self.t = 0

    def update(self, identification):
        if identification is not None:
            non_voc = []
            for i in range(self.n_ag):
                if i == identification:
                    self.presence[i] += 0.3
                    self.last_time_identified[i] = deepcopy(self.t)
                else:
                    if self.t - self.last_time_identified[i] >= self.onset:
                        self.presence[i] -= 0.1 * self.presence[i]
                    non_voc.append(i)
        else:
            for i in range(self.n_ag):
                if self.t - self.last_time_identified[i] >= self.onset:
                    self.presence[i] -= 0.1 * self.presence[i]

        self.presence[self.presence > 1] = 1
        self.presence[self.presence < 0] = 0
        self.t += 1


class ValueEstimation(object):
    def __init__(self, ndims, lr=0.05, discount=0.9):
        self.lr = lr
        self.lr_alpha = 0.1
        self.discount = discount
        self.weights = self.lr * randn(ndims + 1)
        self.prev_state = (1. / ndims) * ones(ndims)
        self.t = 0

    def update(self, state):
        self.lr = 0.1 * (self.t + 1) ** (- self.lr_alpha)
        self.reward = product(state)
        value_current_state = self.weights.dot(hstack((1., state)).T)
        self.td_error = self.reward + self.discount * value_current_state - self.weights.dot(hstack((1., self.prev_state)).T)
        self.weights += self.lr * self.td_error * hstack((1., self.prev_state))
        self.prev_state = deepcopy(state)
        self.t += 1


class ActionSelection(object):
    def __init__(self, ndims):
        self.lr = 0.05  # learning rate
        self.lr_alpha = 0.1
        self.weights = self.lr * randn(ndims + 1)
        self.t = 0

    def decide(self, state):
        activation = self.weights.dot(hstack((1., state)).T)
        self.prev_state = deepcopy(state)
        return activation

    def update(self, td_error, last_action):
        self.lr = 0.1 * (self.t + 1) ** (- self.lr_alpha)
        self.weights += self.lr * td_error * hstack((1., self.prev_state)) * (-1 if last_action is None else 1)
        self.t += 1


class MotorExecution(object):
    def __init__(self, ag_voc_param):
        self.activation_fun = lambda a: (1. + tanh(a * 1.)) / 2.
        self.mean, self.covar = array(ag_voc_param[1]), array(ag_voc_param[2])
        self.m_hist = deque([0] * 5, maxlen=5)
        self.t_last_voc = 0
        self.t = 0

    def execute(self, activation):
        self.activation = self.activation_fun(activation)
        if self.t >= self.t_last_voc + 0 and self.activation > rand():
            self.m = sample_gaussian(self.mean, self.covar)
            self.t_last_voc = deepcopy(self.t)
        else:
            self.m = None
        self.t += 1
        return self.m


class Reflex(object):
    def __init__(self):
        self.t = 0
        self.t_last_amp = 0.
        self.homeo_range = [0.2, 0.8]
        self.level = 0.5
        self.last_act = 0
        self.start_stop = 0

    def activation(self, amp):

        # return 0.

        # if amp > 0.5:
        #     self.start_stop = deepcopy(self.t)
        # if self.t > self.start_stop + 3:
        #     act = 1.
        # else:
        #     act = -1.
        # self.t += 1
        # return act


        if amp > 0.1:
            self.level += 0.3
        else:
            self.level -= 0.1

        if self.level < self.homeo_range[0]:
            self.last_act = 1.
            return 1.
        if self.level > self.homeo_range[1]:
            self.last_act = -3.
            return -3.
        return self.last_act

        return ((1. - min(state)) - 0.5) * 2.

        if min(state) <= 0.5:
            return 1
        else:
            return -1.

        if amp > 0.5:
            self.t_last_amp = deepcopy(self.t)
        if self.t_last_amp > self.t - 5:
            act = -1.
        else:
            act = 0.
        self.t += 1
        return act
        # if amp >= 0.6:
        #     return -2.
        # else:
        #     return 2.
        # if amp <= 0.3:
        #     return 2.
        # elif amp < 0.7:
        #     return 0.
        # else:
        #     return -2.

        # return -2. if amp > 0.6 else 2.


class ModularAgent(Observable):
    def __init__(self, ag_voc_params, id):
        Observable.__init__(self)
        self.id = id
        n_ag = len(ag_voc_params)
        self.ag_voc_param = ag_voc_params[id]
        self.feat_extractor = FeatureExtraction()
        self.identificator = AgentIndentification(ag_voc_params)
        self.pres_estimator = PresenceEstimation(n_ag)
        self.val_estimator = ValueEstimation(n_ag)
        self.decision_maker = ActionSelection(n_ag)
        self.reflex = Reflex()
        self.motor = MotorExecution(self.ag_voc_param)
        self.amp = 0
        self.comfort = True

    def produce(self):
        state = self.pres_estimator.presence
        if min(state) < 0.7:
            self.comfort = False
            activation = self.decision_maker.decide(state)
            activation += 1.
        else:
            self.comfort = True
            activation = -3.
        # activation += self.reflex.activation(self.amp)
        self.m = self.motor.execute(activation)
        self.emit("motor", (self.id, self.m))
        self.emit("activation", (self.id, self.motor.activation))
        return self.m

    def perceive(self, signal):
        s = self.feat_extractor.features(signal)
        self.amp = self.feat_extractor.amplitude(signal)
        percept = self.identificator.identify(s)
        self.pres_estimator.update(percept)
        if not self.comfort:
            self.val_estimator.update(self.pres_estimator.presence)
            self.decision_maker.update(self.val_estimator.td_error, self.m)
        self.emit("presence", (self.id, deepcopy(self.pres_estimator.presence)))
        self.emit("td_error", (self.id, self.val_estimator.td_error))
        self.emit("reward", (self.id, self.val_estimator.reward))
        self.emit("weights", (self.id, self.decision_maker.weights))
        self.emit("amp", (self.id, self.amp))



class Agent(Observable):
    def __init__(self, id, mean, covar, gmm):
        Observable.__init__(self)
        self.id = id
        self.n_ag = gmm.n_components
        self.mean, self.covar = mean, covar
        self.gmm = gmm
        self.presence = hstack((1., (1. / self.n_ag) * ones(self.n_ag)))
        self.lr = 0.05  # learning rate
        self.weights = self.lr * randn(self.n_ag + 1) # array([0.0] * (2 * self.n_ag + 1))
        self.value_weights = self.lr * randn(self.n_ag + 1)
        self.discount = 0.9
        self.last_time_identified = [0] * self.n_ag
        self.t = 0
        self.lr_alpha = 0.1 #99  # decaying lr(t) = t ** (-alpha) according to Damas' IML algorithm
                              # Check if we have sum_t lr(t) == inf and sum_t lr(t)**2 < inf (Eq 21, 22 in Grondman et al.IEEE SMC)
        # self.topics = ["motor", "presence", "td_error", "activation", "weights"]
        self.drive_base = 1.
        self.aud_act = 0.
        self.adapt = True
        self.learn = True
        self.force_voc_each = 0
        self.force_voc_seq = []
        self.last_voc = 0

    def produce(self):
        self.activation = self.drive_base - 2. * self.aud_act
        if self.adapt:
            self.activation = self.weights.dot(self.presence.T)
        self.activation = (1. + tanh(self.activation * 1.)) / 2.  # in [0, 1]
        if self.force_voc_each:
            if not self.t % self.force_voc_each:
                self.activation = 1.
            else:
                self.activation = 0.
        if self.force_voc_seq:
            self.activation = 10. if self.force_voc_seq[self.t % len(self.force_voc_seq)] else -10.

        if self.activation > rand():
            self.m = sample_gaussian(self.mean, self.covar)
        else:
            self.m = None
        self.emit("motor", (self.id, self.m))
        self.emit("activation", (self.id, self.activation))
        return self.m

    def perceive(self, s):
        self.lr = 0.1 * (self.t + 1) ** (- self.lr_alpha)

        bool_ags = [ss is not None for ss in s]
        voc_ag_id = [i for i, elem in enumerate(bool_ags) if elem]
        n_voc_ag = len(voc_ag_id)
        if n_voc_ag > 0:
            self.last_voc = self.t

        prev_presence = deepcopy(self.presence)
        if n_voc_ag == 1:
            probas = self.gmm.predict_proba(s[voc_ag_id[0]])
            ag_id = argmax(probas)
            non_voc = []
            for i in range(self.n_ag):
                if i == ag_id:
                    self.presence[i + 1] += 0.3
                    self.last_time_identified[i] = deepcopy(self.t)
                else:
                    if self.t - self.last_time_identified[i] >= 0:
                        self.presence[i + 1] -= 0.1 * self.presence[i + 1]
                    non_voc.append(i)
        else:
            for i in range(self.n_ag):
                if self.t - self.last_time_identified[i] >= 0:
                    self.presence[i + 1] -= 0.1 * self.presence[i + 1]

        self.presence[self.presence > 1] = 1
        self.presence[self.presence < 0] = 0

        self.aud_act = float(self.t - self.last_voc < 5)

        reward = product(self.presence[1:]) #  - 0.5 * (0 if self.m is None else 1)
        if reward <0: reward = 0

        value_prev_state = self.value_weights.dot(prev_presence.T)
        value_current_state = self.value_weights.dot(self.presence.T)

        td_error = reward + self.discount * value_current_state - value_prev_state

        if self.learn:
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
        self.emit("reward", (self.id, reward))
        self.emit("weights", (self.id, self.weights))
        self.emit("aud_act", (self.id, self.aud_act))

