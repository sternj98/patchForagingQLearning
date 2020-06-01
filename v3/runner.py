import random
from numpy import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import pytz
import datetime
import os
import pandas as pd
import seaborn as sns
import sys
# custom imports
from QLearner import RLInterface
from patchForagingAgents import Model1Agent,Model2Agent,Model3Agent,OmniscientAgent
from patchyEnvironment import PatchEnvironment

STAY = 0
LEAVE = 1

OFF = 0
ON = 1

ITI_state = {"patch" : OFF , "rewindex" : -1}

# run RL interface
env = PatchEnvironment('probabilistic',nTimestates = 50,ITI_penalty = 2,timecost = .2)

agent_type = sys.argv[1]
nTrials = int(sys.argv[2])

# maaybe dynamic beta would help with training here
if agent_type == '1':
    agent = Model1Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,beta = 1.5)
if agent_type == '2':
    agent = Model2Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,beta = 1.5)
if agent_type == '3':
    agent = Model3Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,b = 1,a = 3,beta = 1.5)
if agent_type == 'O':
    agent = OmniscientAgent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,b = 1,a = 3,beta = 1.5)

rl = RLInterface(agent,env)

rl.run_trials(nTrials)

# --------- visualization --------- #
rl.show_qtable()

vis_start = 400
filter_sd = 50

rl.plot_prts(filter_sd)

# assess agent efficiency
rl.plot_rewrate(filter_sd,agent_type)

# mean PRT analysis
rl.prt_bars(filter_sd)
rl.prt_hist(vis_start)
rl.prt_plus_bars(vis_start)

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
