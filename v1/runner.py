# Run Q learning simulations
import random
from numpy import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import pytz
import datetime
import os

from QLearner import RLInterface
from patchForagingAgents import OmniscentAgent,RewSizeAgent,TotalRewAgent
from patchyEnvironment import PatchEnvironment

# run RL interface
env = PatchEnvironment('probabilistic',ITI_len = 7)

agent_type = 'Total Reward'
if agent_type == 'Reward Size':
    agent = RewSizeAgent(env.nTimestates,env.rewsizes,env.ITI_len,epsilon = .90,baseline_epsilon = .1)
if agent_type == 'Total Reward':
    agent = TotalRewAgent(env.nTimestates,max(env.rewsizes),env.ITI_len,epsilon = .90,baseline_epsilon = .1)
if agent_type == 'Omniscient':
    agent = OmniscentAgent(env.nTimestates,env.rewsizes,env.N0,env.ITI_len,epsilon = .90,baseline_epsilon = .1)

rl = RLInterface(agent,env)
print(rl.agent.epsilon)

# should add something so epsilon never goes to exactly 0
decisions = rl.run_trials(100000,epsilon_decay = 3000,record_decisions = True)

# # visualization
rl.show_qtable()
# rl.plot_percent_stay(decisions)
rl.plot_prts(100)

# assess agent efficiency
rl.plot_rewrate(10000,agent_type)

# mean PRT analysis
rl.prt_bars(10000)
rl.prt_hist(10000)
rl.prt_plus_bars(10000)
print(np.mean(rl.prts_plus[1][.125]))
print(np.mean(rl.prts_plus[1][.25]))
print(np.mean(rl.prts_plus[1][.5]))

# timecourse analysis
rl.mk_timecourse()
rl.plot_survival()

# make a function for plotting color density as percent patch residence
rl.percent_hmap([0,10],[20000,30000])

# heatmap rpes
# from behavioral heatmap, how can we evaluate level of exploration vs exploitation in animal behavioral model
rl.rpe_hmap([0,10],[20000,30000])

# heatmap of value
rl.value_hmap([0,10],[20000,30000])
