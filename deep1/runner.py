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
import sys

STAY = 0
LEAVE = 1

OFF = 0
ON = 1

ITI_state = {"patch" : OFF , "rewindex" : -1}

from QLearner import RLInterface
from patchForagingAgents import Model1Agent,Model2Agent,Model3Agent
from patchyEnvironment import PatchEnvironment

# run RL interface
env = PatchEnvironment('probabilistic',nTimestates = 50,ITI_penalty = 2,timecost = .2)

agent_type = sys.argv[1]

# maaybe dynamic beta would help with training here
if agent_type == 'Model1':
    agent = Model1Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,beta = 1.5)
if agent_type == 'Model2':
    agent = Model2Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,beta = 1.5)
if agent_type == 'Model3': # model 3 biases towards longer prts... why?
    agent = Model3Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,b = 1,a = 3,beta = 1.5)

rl = RLInterface(agent,env)

# should add something so epsilon never goes to exactly 0
rl.run_trials(1000,epsilon_decay = 3000)

# # visualization
rl.show_qtable()
# rl.plot_percent_stay(decisions)
rl.plot_prts(10)

# assess agent efficiency
rl.plot_rewrate(10,agent_type)

# mean PRT analysis
rl.prt_bars(10)
# rl.prt_hist(10000)
rl.prt_plus_bars(10)
# print(np.mean(rl.prts_plus[1][.125]))
# print(np.mean(rl.prts_plus[1][.25]))
# print(np.mean(rl.prts_plus[1][.5]))

# timecourse analysis
rl.mk_timecourse()
rl.plot_survival()

# # make a function for plotting color density as percent patch residence
# rl.percent_hmap([0,10],[20000,30000])

# # heatmap rpes
# # from behavioral heatmap, how can we evaluate level of exploration vs exploitation in animal behavioral model
# rl.rpe_hmap([0,10],[20000,30000])

# # heatmap of value
# rl.value_hmap([0,10],[20000,30000])

# plt.figure()
# # sns.distplot(rl.agent.qdiff_test)
# # plt.title('distribution of q value differences')
plt.show()
