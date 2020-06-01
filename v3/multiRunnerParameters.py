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
import progressbar
# custom imports
from QLearner import RLInterface
from patchForagingAgents import Model1Agent,Model2Agent,Model3Agent,OmniscientAgent
from patchyEnvironment import PatchEnvironment

# 2 plots:
#   1. reward over time across parameter settings
#   2. lineplots of fit N0 slope PRT over time
#       - ANOVA score? does ANOVA incr monotonically? or fisher exact test
# immediate 2 parameters of interest: dynamic alpha and beta

# run RL interface
env = PatchEnvironment('probabilistic',nTimestates = 50,ITI_penalty = 2,timecost = 0)

agent_types = sys.argv[1]
nTrials = int(sys.argv[2])
nRepeats = int(sys.argv[3])
resolution = 50 # trials to avg over

# random_control = True
omniscient_control = True

prt_df = pd.DataFrame(columns = ["agent","trial","rewsize","N0","rewsizeN0","PRT"])

widgets = [progressbar.Percentage(), progressbar.Bar()]
bar = progressbar.ProgressBar(widgets=widgets,maxval=nRepeats).start()

if omniscient_control == True:
    for i in range(nRepeats):
        bar.update(i)
        agent = OmniscientAgent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,epsilon = 1,b = 1,a = 3,beta = 1.5)
        rl = RLInterface(agent,env)
        rl.run_trials(nTrials)

        resolution_trials = [resolution * round(x / resolution) for x in list(range(len(rl.rews)))]

        agent_list = ["$%s$" % "Omniscient" for i in range(len(rl.rews))]

        performance_array = np.array([resolution_timepoints,rl.rews,agent_list]).T
        rew_df = rew_df.append(pd.DataFrame(performance_array,columns = ["timepoint","reward","agent"]),sort = True)
    print('\n')

widgets = [progressbar.Percentage(), progressbar.Bar()]
bar = progressbar.ProgressBar(widgets=widgets,maxval=nRepeats).start()

if random_control == True:
    for i in range(nRepeats):
        bar.update(i)
        agent = Model3Agent(len(env.rewsizes),"egreedy",env.nTimestates,env.rewsizes,epsilon = 1,b = 1,a = 3,beta = 1.5)
        rl = RLInterface(agent,env)
        rl.run_trials(nTrials)
        resolution_timepoints = [resolution * round(x / resolution) for x in list(range(len(rl.rews)))]

        agent_list = ["$%s$" % "Random" for i in range(len(rl.rews))]

        performance_array = np.array([resolution_timepoints,rl.rews,agent_list]).T
        rew_df = rew_df.append(pd.DataFrame(performance_array,columns = ["timepoint","reward","agent"]),sort = True)
    print('\n')
