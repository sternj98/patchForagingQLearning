import random
from numpy import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import pytz
import datetime
import os
import pandas as pd
import itertools

STAY = 0
LEAVE = 1

OFF = 0
ON = 1

class Agent:
    """
    General class for Q learning agents
    """
    def __init__(self,nActions,integration_dim,num_timesteps,ITI_len,epsilon = .5,baseline_epsilon = 0,omniscient = False):
        self.Q = [np.zeros((nActions,ITI_len)),np.zeros((nActions,integration_dim,num_timesteps))]
        self.epsilon0 = epsilon
        self.epsilon = epsilon # proportion of exploration
        self.baseline_epsilon = baseline_epsilon
        self.Lambda = 0.8 # eligibility trace decay
        self.lr = 0.1 # learning rate
        self.discount = .7 #.9
        self.omniscient = omniscient

    def select_action(self,internal_state):
        """
            Agent chooses an action
            Returns: new action
        """
        if rnd.random() > self.epsilon:
            return self.greedy_action(internal_state)
        else:
            return self.random_action()

    def random_action(self):
        """
            Agent takes a random action
        """
        return STAY if random.random() < 0.5 else LEAVE

    def greedy_action(self,internal_state):
        """
            Agent takes most rewarding action in current state according to Q table
        """
#         print(internal_state[1:])
        if self.Q[internal_state[0]][(STAY,)+internal_state[1:]] > self.Q[internal_state[0]][(LEAVE,)+internal_state[1:]]:
            return STAY
        # Is LEAVE reward bigger?
        elif self.Q[internal_state[0]][(LEAVE,)+internal_state[1:]] > self.Q[internal_state[0]][(STAY,)+internal_state[1:]]:
            return LEAVE
        # Rewards are equal, take random action
        return STAY if random.random() < 0.5 else LEAVE

    def update(self, old_state, new_state, action, reward):
        """
            Update agent Q-table based on experience
            Arguments: old_state,new_state,action,reward
        """

        q_old = self.Q[old_state[0]][(action,)+old_state[1:]] # Old Q-table value
        future_action = self.greedy_action(new_state) # Select next best action
        #  print('future_action:',future_action)
        EV_new = self.Q[new_state[0]][(future_action,)+new_state[1:]] # What is reward for the best next action?

        # Main Q-table updating algorithm
#         new_value = q_old + self.lr * (reward + self.discount * EV_new - q_old) # add lambda here
#         print(self.Q[action][old_state])
#         self.Q[action][old_state] = new_value

        self.Q[old_state[0]][(action,)+old_state[1:]] += self.lr * (reward + self.discount * EV_new - q_old)

        return reward + self.discount * EV_new - q_old # return rpe now
#         print(self.Q[action][old_state])
#         print('action:',action)
#         testleave2 = self.Q[old_state[0]][(LEAVE,)+old_state[1:]]
#         teststay2 = self.Q[old_state[0]][(STAY,)+old_state[1:]]

#         print('leave:',testleave,'->',testleave2)
#         print('stay:',teststay,'->',teststay2)


class OmniscentAgent(Agent):
    """
        Agent has perfect knowledge of patch reward size and frequency
    """
    def __init__(self,nTimestates,rewsizes,freqs,ITI_len,epsilon = .5,baseline_epsilon = .1):
        nActions = 2
        nPatches = int(len(rewsizes) * len(freqs))
        super().__init__(nActions,nPatches,nTimestates,ITI_len,epsilon = epsilon,baseline_epsilon = baseline_epsilon,omniscient = True)
        self.rewsizes = rewsizes
        self.freqs = freqs
        self.patches = list(itertools.product([0,1,2], repeat = 2))
        self.curr_patch = 0

    def internalize_state(self,env_state):
        """
            grab the patch and time spent on patch
            returns internal_state as a tuple for indexing q table
        """
        if env_state[0] == ON:
            internal_state = tuple((ON,self.curr_patch,env_state[1]))
        else:
            internal_state = tuple(env_state)
        return internal_state

    def integrate(self,curr_rewsize,curr_freq):
        """
            Returns a patch between 0 and 8
        """
        self.curr_patch = self.patches.index((self.rewsizes.index(curr_rewsize),self.freqs.index(curr_freq)))

    def reset_integration(self):
        self.curr_patch = 0

class RewSizeAgent(Agent):
    """
        Agent processes only the reward size in the q table
    """
    def __init__(self,nTimestates,rewsizes,ITI_len,epsilon = .5,baseline_epsilon = .1):
        nActions = 2
        super().__init__(nActions,len(rewsizes),nTimestates,ITI_len,epsilon = epsilon,baseline_epsilon = baseline_epsilon)
        self.curr_rewsize = 0
        self.rewsizes = rewsizes

    def internalize_state(self,env_state):
        """
            Use agent memory of the current reward size and env information on time
            returns internal_state = (curr_rewsize,time on patch) as a tuple for array indexing
        """
        if env_state[0] == ON:
            internal_state = tuple((ON,self.rewsizes.index(self.curr_rewsize),env_state[1]))
        else:
            internal_state = tuple(env_state)
        return internal_state

    def integrate(self,rew):
        self.curr_rewsize = rew

    def reset_integration(self):
        self.curr_rewsize = 0

class TotalRewAgent(Agent):
    """
        Agent has memory of rewards received and processes this as part of Q table
    """
    def __init__(self,nTimestates,max_rewsize,ITI_len,epsilon = .5,baseline_epsilon = .1):
        max_totalrew = nTimestates * max_rewsize
        print(max_totalrew)
        nActions = 2
        super().__init__(nActions,max_totalrew,nTimestates,ITI_len,epsilon = epsilon,baseline_epsilon = baseline_epsilon)
        self.elg_states = []
        self.curr_totalrew = 0

    def internalize_state(self,env_state):
        """
            Use agent memory of total rew received on current patch and env information on time
            returns internal_state = (curr_totalrew,time on patch)
        """
        if env_state[0] == ON:
            internal_state = tuple((ON,self.curr_totalrew,env_state[1]))
        else:
            internal_state = tuple(env_state)
        return internal_state

    def integrate(self,rew):
        self.curr_totalrew += int(rew)

    def reset_integration(self):
        self.curr_totalrew = 0
