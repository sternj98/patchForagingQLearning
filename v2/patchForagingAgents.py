import random
from numpy import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import pytz
import datetime
import os
import pandas as pd
import itertools
import seaborn as sns

STAY = 0
LEAVE = 1

OFF = 0
ON = 1

ITI_state = {"patch" : OFF , "rewindex" : -1}


class Agent:
    """
    General class for Q learning agents
    """
    def __init__(self,nRewSizes,integration_dim,decision_type,rewsizes,beta = 1,epsilon = .9,baseline_epsilon = 0.1,lr = .1 ):
        self.Q = [np.zeros(2),np.zeros((nRewSizes,2,integration_dim))]
        self.epsilon0 = epsilon
        self.epsilon = epsilon # proportion of exploration in egreedy
        self.baseline_epsilon = baseline_epsilon # baseline exploration for egreedy
        self.beta = beta
        self.lr = lr # learning rate
        self.discount = .8 #.9
        self.decision_type = decision_type # "egreedy" or "softmax"
        self.nRewSizes = nRewSizes
        self.rewsizes = rewsizes
        self.test1 = {1 : [] , 2 : [] , 4 : []}
        self.test2 = {1 : [] , 2 : [] , 4 : []}

    def select_action(self,rewsize_index,rew_int,patch):
        """
            Agent chooses an action
            Returns: new action
        """
        if self.decision_type == "egreedy":
            if rnd.random() > self.epsilon:
                return self.greedy_action(rewsize_index,rew_int,patch)
            else:
                return self.random_action()
        elif self.decision_type == "softmax":
            if patch == ON:
                Q_stay = self.Q[ON][rewsize_index,STAY,rew_int]
                Q_leave = self.Q[ON][rewsize_index,LEAVE,rew_int]
            else:
                Q_stay = self.Q[OFF][STAY]
                Q_leave = self.Q[OFF][LEAVE]
            p_stay = (1 + np.exp(-self.beta * (Q_stay - Q_leave))) ** (-1)
            return STAY if rnd.rand() < p_stay else LEAVE
        else:
            raise ValueError("Please use \"egreedy\" or \"softmax\" as decision type")

    def random_action(self):
        """
            Agent takes a random action
        """
        return STAY if random.random() < 0.5 else LEAVE

    def greedy_action(self,rewsize_index,rew_int,patch):
        """
            Agent takes most rewarding action in current state according to Q table
        """
        if patch == ON:
            Q_stay = self.Q[ON][rewsize_index,STAY,rew_int]
            Q_leave = self.Q[ON][rewsize_index,LEAVE,rew_int]
        else:
            Q_stay = self.Q[OFF][STAY]
            Q_leave = self.Q[OFF][LEAVE]
        if Q_stay > Q_leave:
            return STAY
        elif Q_leave > Q_stay:
            return LEAVE
        return STAY if random.random() < 0.5 else LEAVE # Rewards are equal, take random action

    def update(self,old_rewsize_index, old_rew_int, patch_old,
                    new_rewsize_index, new_rew_int, patch_new,
                    action, reward):
        """
            Update agent Q-table based on experience
            Arguments: old_state,new_state,action,reward
        """
        if patch_old == ON and patch_new == ON:
            q_old = self.Q[ON][old_rewsize_index,action,old_rew_int] # Old Q-table value
            future_action = self.greedy_action(new_rewsize_index,new_rew_int,ON) # Select next best action
            EV_new = self.Q[ON][new_rewsize_index,future_action,new_rew_int]
            rpe = reward + self.discount * EV_new - q_old
            self.Q[ON][old_rewsize_index,action,old_rew_int] += self.lr * rpe

        elif patch_old == ON and patch_new == OFF:
            q_old = self.Q[ON][old_rewsize_index,action,old_rew_int] # Old Q-table value
            future_action = self.greedy_action(new_rewsize_index,new_rew_int,OFF)
            EV_new = self.Q[OFF][future_action]
            rpe = reward + self.discount * EV_new - q_old
            self.Q[ON][old_rewsize_index,action,old_rew_int] += self.lr * rpe

        elif patch_old == OFF and patch_new == ON:
            q_old = self.Q[OFF][action]
            future_action = self.greedy_action(new_rewsize_index,new_rew_int,ON) # Select next best action
            EV_new = self.Q[ON][new_rewsize_index,future_action,new_rew_int]
            rpe = reward + self.discount * EV_new - q_old
            self.Q[OFF][action] += self.lr * rpe

        elif patch_old == OFF and patch_new == OFF:
            q_old = self.Q[OFF][action]
            future_action = self.greedy_action(new_rewsize_index,new_rew_int,OFF)
            EV_new = self.Q[OFF][future_action]
            rpe = reward + self.discount * EV_new - q_old
            self.Q[OFF][action] += self.lr * rpe

        return rpe

class Model1Agent(Agent):
    """
        rew integration is a function of time
    """
    def __init__(self,nRewSizes,decision_type,nTimestates,rewsizes,
                                              beta = 1,epsilon = .5,baseline_epsilon = .1,lr = .1 ):
        integration_dim = nTimestates
        super().__init__(nRewSizes,integration_dim,decision_type,rewsizes,
                                              beta = beta,epsilon = epsilon,baseline_epsilon = baseline_epsilon,lr =lr)
        self.model = "Model1"
    def integrate(self,env_state):
        if env_state["patch"] == ON:
            return env_state["t"]
        else:
            return -1

class Model2Agent(Agent):
    """
        rew integration is a function of time since previous reward, reward size
    """
    def __init__(self,nRewSizes,decision_type,nTimestates,rewsizes,
                                              beta = 1,epsilon = .5,baseline_epsilon = .1,lr = .1):
        integration_dim = nTimestates
        super().__init__(nRewSizes,integration_dim,decision_type,rewsizes,
                                              beta = beta,epsilon = epsilon,baseline_epsilon = baseline_epsilon,lr = lr)
        self.model = "Model2"
    def integrate(self,env_state):
        if env_state["patch"] == ON:
            # print(env_state["rews"][:env_state["t"]])
            time_since = list(reversed(env_state["rews"][:(env_state["t"]+1)])).index(env_state["rewsize"])
            # print(time_since)
            return time_since
        else:
            return -1

class Model3Agent(Agent):
    """
        rew integration is a function of total rewards received over time, reward size
    """
    def __init__(self,nRewSizes,decision_type,nTimestates,rewsizes,
                                            a = 2,b = 1,
                                            beta = 1,epsilon = .5,baseline_epsilon = .1,lr = .1):
        integration_dim = 2 * nTimestates * a # so we never go negative in Q indexing
        super().__init__(nRewSizes,integration_dim,decision_type,rewsizes,
                                   beta=beta,epsilon = epsilon,baseline_epsilon = baseline_epsilon,lr = lr)
        self.a = a
        self.b = b
        self.integration_baseline = integration_dim / 2
        self.model = "Model3"
#         print(self.integration_baseline)

    def integrate(self,env_state):
        if env_state["patch"] == ON:
            t = env_state["t"]
            rew_int = self.a * sum(env_state["rews"][:(t+1)])/env_state["rewsize"] - self.b * t + self.integration_baseline
            return int(rew_int)
        else:
            return -1

# class Model4Agent(Agent):
#     """
#         use Jan's model for posterior mean:
#         E[n0|Z(t)] = (a0 + Z(t)) / (b0 + tau(1 - e^{-t/tau}))
#     """
#     def __init__(self,nRewSizes,decision_type,nTimestates,rewsizes,
#                                             a = 2,b = 1,
#                                             beta = 1,epsilon = .5,baseline_epsilon = .1,lr = .1):
#         integration_dim = 2 * nTimestates * a # so we never go negative in Q indexing
#         super().__init__(nRewSizes,integration_dim,decision_type,rewsizes,
#                                    beta=beta,epsilon = epsilon,baseline_epsilon = baseline_epsilon,lr = lr)
#         self.a = a
#         self.b = b
#         self.integration_baseline = integration_dim / 2
#         self.model = "Model4"
# #         print(self.integration_baseline)
#
#     def integrate(self,env_state):
#         if env_state["patch"] == ON:
#             t = env_state["t"]
#             rew_int = self.a * sum(env_state["rews"][:(t+1)])/env_state["rewsize"] - self.b * t + self.integration_baseline
#             return int(rew_int)
#         else:
#             return -1

# how to get recency bias in here... does model 3 have this already?
