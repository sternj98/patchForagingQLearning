import numpy as np
from numpy import random as rnd
from collections import deque

STAY = 0
LEAVE = 1

class MVT_agentAverager:
    """
        Class for direct MVT-approximation agents
        From Table 2 of Daw and Constantino 2015

        avg (rho): longrun environment reward rate (dynamic estimation)
        h: time it takes to harvest (= 1 here)
        s: expected rew from next timestep
        inst (kappa): estimated env depletion rate (= .125 here)

        Problem: inst rew rate can be 0
            - better to do this also using a delta rule, initialized at avg value
            --> estimate inst with 3-timestep sliding avg, where empty entries
                are replaced with avg
            - improve w/ more advanced inference?
    """
    def __init__(self,ITI_penalty,c = .5,lr = .001,tau = .1, beta = 2.0,kappa = 1,recent_rews_len = 5):
        # parameters
        self.kappa = kappa
        self.lr = lr # LR for env rewrate
        self.beta = beta # inverse temperature
        self.c = c # stay bias term, should be able to drift over time
        self.ITI_penalty = ITI_penalty # to determine when we are on patch

        # initialize estimation quantities
        self.avg = 0
        self.inst = self.avg
        self.recent_rews = deque([self.avg for i in range(recent_rews_len)])
        self.recent_rews_len = recent_rews_len
        self.newTrial = False

        # initialize datastructures
        self.avg_list = []
        self.inst_list = []

    def select_action(self):
        """
            Agent chooses an action
            Returns: new action
        """
        p_stay = 1 / (1 + np.exp(-self.c - self.beta * (self.inst * self.kappa - self.avg)))
        if rnd.rand() < p_stay:
            return STAY
        else:
            return LEAVE

    def update(self,rew):
        """
            Update avg env rew (rho) according to delta rule
        """
        # update recent reward memory only if on patch
        if rew <= -self.ITI_penalty:
            self.inst = self.inst # don't update in ITI
            self.newTrial = True
            self.inst_list.append(self.inst)
        elif self.newTrial == True:
            self.newTrial = False
            self.recent_rews = deque([rew]) # initialize to reward received in beginning of tria
            self.inst = np.mean(self.recent_rews)
            self.inst_list.append(self.inst)
        else:
            self.recent_rews.append(rew)
            if len(self.recent_rews) > self.recent_rews_len:
                self.recent_rews.popleft()
            self.inst = np.mean(self.recent_rews)  # use sliding avg to update inst if on patch
            self.inst_list.append(self.inst)

        # update longterm reward estimation
        delta = rew - self.avg
        self.avg = (1 - self.lr) * self.avg + self.lr * delta
        self.avg_list.append(self.avg)



#### CHANGE THIS TO MONTE CARLO TD (just take out the (1 - instLR)) ####
class MVT_agentDoubleDelta:
    """
        Class for direct MVT-approximation agents
        From Table 2 of Daw and Constantino 2015

        avg (rho): longrun environment reward rate (dynamic estimation)
        h: time it takes to harvest (= 1 here)
        s: expected rew from next timestep
        inst (kappa): estimated env depletion rate

        Estimate instantaneous rewrate on patch with delta rule
    """
    # LR
    def __init__(self,ITI_penalty,c = 0.,lr = .001,lrInst = .25, beta = 2.0,kappa = .8):
        # parameters
        self.ITI_penalty = ITI_penalty # just to monitor where we are
        self.lr = lr # LR for env rewrate
        self.lrInst = lrInst # LR for inst rewrate
        self.beta = beta # inverse temperature
        self.c = c # stay bias term, should be able to drift over time
        self.kappa = kappa

        # initialize estimation quantities
        self.avg = 0
        self.inst = self.avg
        self.newTrial = False

        # initialize datastructures
        self.avg_list = []
        self.inst_list = []

    def select_action(self):
        """
            Agent chooses an action
            Returns: new action
        """
        p_stay = 1 / (1 + np.exp(-self.c - self.beta * (self.inst * self.kappa - self.avg)))

        if rnd.rand() < p_stay:
            return STAY
        else:
            return LEAVE

    def update(self,rew):
        """
            Update avg env rew (rho) according to delta rule
        """
        # initialize as the first reward received, then do a delta rule
        # update recent reward memory only if on patch
        if rew <= -self.ITI_penalty:
            self.inst = self.inst # don't update in ITI
            self.newTrial = True
            self.inst_list.append(self.inst)
        elif self.newTrial == True:
            self.newTrial = False
            self.inst = rew # initialize to reward received in beginning of trial
            self.inst_list.append(self.inst)
        else:
            delta = rew - self.inst
            self.inst = (1 - self.lrInst) * self.inst + self.lrInst * delta  # update using delta rule
            self.inst_list.append(self.inst)

        # update env rewrate estimation
        delta = rew - self.avg
        self.avg = (1 - self.lr) * self.avg + self.lr * delta
        self.avg_list.append(self.avg)
