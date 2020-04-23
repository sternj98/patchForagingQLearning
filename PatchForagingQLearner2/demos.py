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

# first: demonstrate learning at the timestep scale: step learning through trials

env = PatchEnvironment('probabilistic',nTimestates = 50,ITI_penalty = 2,timecost = 0)

agent_type = 1

sim = 2

if sim == 1:
    if agent_type == 1:
        agent = Model1Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,beta = 5,lr = .25)
    elif agent_type ==2:
        agent = Model2Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,beta = 5,lr = .5)
    elif agent_type == 3:
        agent = Model3Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,b = 1,a = 3,beta = 5,lr = .5)

if sim == 2:
    if agent_type == 1:
        agent = Model1Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,beta = 1.4,lr = .2)
    elif agent_type ==2:
        agent = Model2Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,beta = 1,lr = .5)
    elif agent_type == 3:
        agent = Model3Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,b = 1,a = 3,beta = 5,lr = .5)

rl1 = RLInterface(agent,env)

# use n0 values here to visualize mean differences in staying time
trialtype1 = {"rews" : [1,0,1,0,1,0,0,0,0,0] , "n0" : .125}
trialtype2 = {"rews" : [1,0,0,0,1,1,1,1,1,1] , "n0" : .25}
trialtype3 = {"rews" : [1,0,0,1,0,1,0,1,0,1,0] , "n0" : .5}
trialtype4 = {"rews" : [4,4,4,4,4,4,4,0,0,0] , "n0" : .125}

tt1_series = [trialtype1 for n in range(20)]
tt4_series = [trialtype4 for n in range(20)]
tt3_series = [trialtype3 for n in range(20)]
tt12_series = [trialtype1 if n < 20 else trialtype2 for n in range(40)]
tt21_series = [trialtype2 if n < 20 else trialtype1 for n in range(20)]

tt14_alt = [trialtype1 if rnd.rand() < .5 else trialtype4 for n in range(100)]

tt14_series = [trialtype1 if n < 40 else trialtype4 for n in range(80)]
tt14_series_alt = tt14_series + tt14_alt

print(tt14_series_alt)

if sim == 2:
    rl1.run_trials(40,probe_specs = tt1_series + tt4_series + tt14_alt )# tt1_series + tt1_series + tt4_series + tt14_alt)
    rl1.barcode_beh()
    rl1.show_qtable()
    rl1.plot_qiti()

if sim == 1:

    Qlist = []
    Qarray = []
    qitilist = []
    actions = []
    rews = []
    patchOn = []

    for trial in range(20):
        patchOn.append(rl1.env.state["patch"])
        action,rew,rpe,value = rl1.step(probe_trial = tt[trial])
        Qlist.append(rl1.agent.Q[ON].copy())
        qitilist.append(rl1.agent.Q[OFF][LEAVE])
        actions.append(action)
        rews.append(rew)

    fig = plt.figure(figsize = (2 * 10 , 2 * 2))
    for trial in range(20):
        plt.subplot(10, 2, trial+1)
        plt.subplots_adjust(hspace = 1.3)
        if agent_type == 1:
            sns.heatmap(Qlist[trial][0 , :, :7],cbar_kws=dict(ticks=[-1,0,1,1.5,2]),annot = True)
        if agent_type == 2:
            sns.heatmap(Qlist[trial][0 , :,:10],cbar_kws=dict(ticks=[-1,0,1,1.5,2]),annot = True)
        if agent_type == 3:
            sns.heatmap(Qlist[trial][0 , :, 150:160],cbar_kws=dict(ticks=[-1,0,1,1.5,2]),annot = True)
        if actions[trial] == LEAVE:
            actn = "Leave"
        else:
            actn = "Stay"
        if patchOn[trial] == ON:
            ptch = "On"
        else:
            ptch = "Off"
        plt.xticks([])
        plt.yticks([])

        plt.xlabel("Reward:%i Action:%s Patch:%s" % (rews[trial], actn,ptch))
    if agent_type == 3:
        plt.suptitle('Model3 a=3 b = 1 ; Trial Rewards: [1,0,0,1,0,1,0,1,0,1,0]')
    if agent_type == 2:
        plt.suptitle('Model2 ; Trial Rewards: [1,0,0,1,0,1,0,1,0,1,0]')
    if agent_type == 1:
        plt.suptitle('Model1 ; Trial Rewards: [1,0,0,1,0,1,0,1,0,1,0]')
    plt.figure()
    plt.plot(qitilist)
    plt.title('Q Value for Leave ITI State over time')


# plt.tight_layout()
plt.show()


# second: after learning, show performance on 200 trial timescale
# trial by trial comparison of reward integration
