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

# global function to convert LEAVE Q values to a single state
def leaveOneQ(rewsize_index,action,rew_int):
    if action == LEAVE:
        return (0,LEAVE,0)
    else:
        return (rewsize_index,STAY,rew_int)

class Agent:
    """
    General class for Q learning agents
    """
    def __init__(self,nRewSizes,integration_dim,decision_type,rewsizes,
                 lmda = 0.2, # eligibility trace parameter
                 lr = .1,dynamic_lr = False,lr0 = 1.5,lr_final = .05,lr_decay = 200,
                 epsilon = .9,
                 beta = 1.5,dynamic_beta = False,beta0 = .5,beta_final = 1.5,beta_decay = 200,
                 discount = 0.8):
        self.Q = [np.zeros(2),np.zeros((nRewSizes,2,integration_dim))]
        self.decision_type = decision_type # "egreedy" or "softmax"
        self.dynamic_beta = dynamic_beta
        self.dynamic_lr = dynamic_lr
        if decision_type == "egreedy":
            self.epsilon = epsilon # proportion of exploration in egreedy
        elif decision_type == "softmax":
            self.beta = beta
            if dynamic_beta == True:
                self.beta0 = beta0
                self.beta_final = beta_final
                self.beta_decay = beta_decay
        self.lr = lr # learning rate
        if dynamic_lr == True:
            self.lr0 = lr0
            self.lr_final = lr_final
            self.lr_decay = lr_decay
        self.discount = discount #.9
        self.nRewSizes = nRewSizes
        self.rewsizes = rewsizes
        # eligibility trace parameters
        self.lmda = lmda
        self.elg = [np.zeros(2),np.zeros((nRewSizes,2,integration_dim))] # copy of Q table dims

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
                Q_leave = self.Q[ON][0,LEAVE,0]
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
            Q_leave = self.Q[ON][0,LEAVE,0]
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
        # add trace decay
        self.elg[OFF] = self.elg[OFF] * self.lmda
        self.elg[ON] = self.elg[ON] * self.lmda

        if patch_old == ON and patch_new == ON:
            q_old = self.Q[ON][old_rewsize_index,action,old_rew_int] # Old Q-table value (first actn not leave)
            self.elg[ON][old_rewsize_index,action,old_rew_int] = (1 - self.lmda) # add eligibility
            future_action = self.greedy_action(new_rewsize_index,new_rew_int,ON) # Select next best action
            EV_new = self.Q[ON][leaveOneQ(new_rewsize_index,future_action,new_rew_int)] # in case leave
            rpe = reward + self.discount * EV_new - q_old
            # self.Q[ON][old_rewsize_index,action,old_rew_int] += self.lr * rpe

        elif patch_old == ON and patch_new == OFF:
            q_old = self.Q[ON][leaveOneQ(old_rewsize_index,action,old_rew_int)] # Old Q-table value
            self.elg[ON][leaveOneQ(old_rewsize_index,action,old_rew_int)] = (1 - self.lmda) # add eligibility
            future_action = self.greedy_action(new_rewsize_index,new_rew_int,OFF)
            EV_new = self.Q[OFF][future_action]
            rpe = reward + self.discount * EV_new - q_old
            # self.Q[ON][leaveOneQ(old_rewsize_index,action,old_rew_int)] += self.lr * rpe

        elif patch_old == OFF and patch_new == ON:
            q_old = self.Q[OFF][action]
            self.elg[OFF][action] = (1 - self.lmda) # add eligibility
            future_action = self.greedy_action(new_rewsize_index,new_rew_int,ON) # Select next best action
            EV_new = self.Q[ON][leaveOneQ(new_rewsize_index,future_action,new_rew_int)]
            rpe = reward + self.discount * EV_new - q_old
            # self.Q[OFF][action] += self.lr * rpe

        elif patch_old == OFF and patch_new == OFF:
            q_old = self.Q[OFF][action]
            self.elg[OFF][action] = (1 - self.lmda) # add eligibility
            future_action = self.greedy_action(new_rewsize_index,new_rew_int,OFF)
            EV_new = self.Q[OFF][future_action]
            rpe = reward + self.discount * EV_new - q_old
            # self.Q[OFF][action] += self.lr * rpe

        self.Q[OFF] += self.elg[OFF] * self.lr * rpe
        self.Q[ON] += self.elg[ON] * self.lr * rpe

        return rpe

class Model1Agent(Agent):
    """
        rew integration is a function of time
    """
    def __init__(self,nRewSizes,decision_type,nTimestates,rewsizes,
                                              beta = 1,epsilon = .5,lr = .1 ):
        integration_dim = nTimestates
        super().__init__(nRewSizes,integration_dim,decision_type,rewsizes,
                                              beta = beta,epsilon = epsilon,lr =lr)
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
                                              beta = 1,epsilon = .5,lr = .1):
        integration_dim = nTimestates
        super().__init__(nRewSizes,integration_dim,decision_type,rewsizes,
                                              beta = beta,epsilon = epsilon,lr = lr)
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
                                            a = 3,b = 1,
                                            beta = 1,epsilon = .5,lr = .1):
        integration_dim = 2 * nTimestates * a # so we never go negative in Q indexing
        super().__init__(nRewSizes,integration_dim,decision_type,rewsizes,
                                   beta=beta,epsilon = epsilon,lr = lr)
        self.a = a
        self.b = b
        self.integration_baseline = integration_dim / 2
        self.model = "Model3"
        # print("Integration baseline:",self.integration_baseline)

    def integrate(self,env_state):
        if env_state["patch"] == ON:
            t = env_state["t"]
            rew_int = self.a * sum(env_state["rews"][:(t+1)])/env_state["rewsize"] - self.b * t + self.integration_baseline
            # print(int(rew_int))
            return int(rew_int)
        else:
            return -1

class OmniscientAgent(Agent):
    """
        rew integration is a function of total rewards received over time, reward size
    """
    def __init__(self,nRewSizes,decision_type,nTimestates,rewsizes,
                                            a = 2,b = 1,
                                            beta = 1,epsilon = .5,lr = .1):
        integration_dim = 15 * 3 + 8 # hacky here
        super().__init__(nRewSizes,integration_dim,decision_type,rewsizes,
                                   beta=beta,epsilon = epsilon,lr = lr)

        self.integration_baseline = integration_dim / 2
        self.model = "Omniscient"

    def integrate(self,env_state):
        if env_state["patch"] == ON:
            t = env_state["t"]
            # N0idx = [.125,.25,.5].index(env_state["n0"]) # tough hardcoding but hey
            N0idx = [.125,.25,.5].index(env_state["n0"])
            rew_int = N0idx * 15 + t
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

# how to get recency bias in here...
