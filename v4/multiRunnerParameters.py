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
from RLInterface import RLInterface
from patchForagingAgents import Model1Agent,Model2Agent,Model3Agent,OmniscientAgent
from patchyEnvironment import PatchEnvironment
from performancePlots import plot_avgrew,plot_prt_slopes

# Immediate parameters of interest: dynamic lr and beta, lambda
# Assay convergence with PRT slope plots and reward rate

# run RL interface
env = PatchEnvironment('probabilistic',nTimestates = 50,ITI_penalty = 2,timecost = 0)

# single agent type, range of parameters to run over
agent_type = sys.argv[1]
nTrials = int(sys.argv[2])
nRepeats = int(sys.argv[3])
step_res = 40
trial_res = 100

# define parameter ranges of interest
lr0 = 1.5
lr0_min = 0.5
lr0_max = 2.
lr0_steps = 5
lr0_range = np.linspace(lr0_min,lr0_max,lr0_steps)
lr_decay_min = 50
lr_decay_max = 500
lr_decay_steps = 5
lr_decay_range = np.linspace(lr_decay_min,lr_decay_max,lr_decay_steps)

random_control = False
omniscient_control = True

rew_df = pd.DataFrame(columns = ["timepoint","reward"])
prt_df = pd.DataFrame(columns = ["agent","trial","rewsize","N0","rewsizeN0","PRT"])

widgets = [progressbar.Percentage(), progressbar.Bar()]
bar = progressbar.ProgressBar(widgets=widgets,maxval=nRepeats).start()

if omniscient_control == True:
    for i in range(nRepeats):
        bar.update(i)
        agent = OmniscientAgent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,epsilon = 1,b = 1,a = 3,beta = 1.5)
        rl = RLInterface(agent,env)
        rl.run_trials(nTrials)

        # log rewards
        resolution_timepoints = [step_res * round(x / step_res) for x in list(range(len(rl.rews)))]
        agent_list = ["$%s$" % "Omniscient" for i in range(len(rl.rews))] # just for logging
        performance_array = np.array([resolution_timepoints,rl.rews,agent_list]).T
        rew_df = rew_df.append(pd.DataFrame(performance_array,columns = ["timepoint","reward","agent"]),sort = True)

        # log PRTs
        iPRT_df = rl.prt_df
        resolution_trials = [trial_res * round(x / trial_res) for x in list(range(nTrials))]
        agent_list = ["$%s$" % "Omniscient" for i in range(nTrials)]
        iPRT_df["agent"], iPRT_df["trial"] = agent_list, resolution_trials
        prt_df = prt_df.append(iPRT_df,sort = True)

    print('\n')

widgets = [progressbar.Percentage(), progressbar.Bar()]
bar = progressbar.ProgressBar(widgets=widgets,maxval=nRepeats).start()

if random_control == True:
    for i in range(nRepeats):
        bar.update(i)
        agent = Model3Agent(len(env.rewsizes),"egreedy",env.nTimestates,env.rewsizes,epsilon = 1,b = 1,a = 3,beta = 1.5)
        rl = RLInterface(agent,env)
        rl.run_trials(nTrials)

        # log rewards
        resolution_timepoints = [step_res * round(x / step_res) for x in list(range(len(rl.rews)))]
        agent_list = ["$%s$" % "Random" for i in range(len(rl.rews))]
        performance_array = np.array([resolution_timepoints,rl.rews,agent_list]).T
        rew_df = rew_df.append(pd.DataFrame(performance_array,columns = ["timepoint","reward","agent"]),sort = True)

        # log PRTs
        iPRT_df = rl.prt_df
        resolution_trials = [trial_res * round(x / trial_res) for x in list(range(nTrials))]
        agent_list = ["$%s$" % "Random" for i in range(nTrials)]
        iPRT_df["agent"], iPRT_df["trial"] = agent_list, resolution_trials
        prt_df = prt_df.append(iPRT_df,sort = True)
    print('\n')

widgets = [progressbar.Percentage(), progressbar.Bar()]
bar = progressbar.ProgressBar(widgets=widgets,maxval=nRepeats * 5 * 5).start()

counter = 0

for lr0 in lr0_range:
    for lr_decay in lr_decay_range:
        # Random behavior as a baseline
        for i in range(nRepeats):
            counter += 1
            bar.update(counter)
            if agent_type == 1:
                agent = Model1Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,dynamic_lr = True,lr0 = lr0,lr_decay = lr_decay)
                agent_str = "Model 1"
            if agent_type == 2:
                agent = Model2Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,dynamic_lr = True,lr0 = lr0,lr_decay = lr_decay)
                agent_str = "Model 2"
            if agent_type == 3:
                agent = Model3Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,dynamic_lr = True,lr0 = lr0,lr_decay = lr_decay)
                agent_str = "Model 3"

            rl = RLInterface(agent,env)
            rl.run_trials(nTrials)

            # log rewards
            resolution_timepoints = [step_res * round(x / step_res) for x in list(range(len(rl.rews)))]
            agent_list = ["$lr0:%f-lr_decay:%f$" % (lr0,lr_decay) for i in range(len(rl.rews))]
            performance_array = np.array([resolution_timepoints,rl.rews,agent_list]).T
            rew_df = rew_df.append(pd.DataFrame(performance_array,columns = ["timepoint","reward","agent"]),sort = True)

            # log PRTs
            iPRT_df = rl.prt_df
            resolution_trials = [trial_res * round(x / trial_res) for x in list(range(nTrials))]
            agent_list = ["$lr0:%f-lr_decay:%f$" % (lr0,lr_decay) for i in range(nTrials)]
            iPRT_df["agent"], iPRT_df["trial"] = agent_list, resolution_trials
            prt_df = prt_df.append(iPRT_df,sort = True)

# visualize the reward rate, std across runs
plot_avgrew(rew_df)
plot_prt_slopes(prt_df,rollover = 5)
rl.show_qtable()
plt.show()
print('\n')
