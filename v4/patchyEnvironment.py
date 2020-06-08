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


# Methods to draw from discrete expo distributions for Patch Environment
def cdf(x,N0):
    A = -.125
    return 1+ N0 * 1/A * np.exp(A*x) - N0/A
    
def pdf(this_cdf):
    this_pdf = []
    for t in range(1,len(this_cdf)):
        this_pdf.append(this_cdf[t]-this_cdf[t-1])
    this_pdf.insert(0,1) # add that first reward
    return this_pdf

def generate_pdfs(N0):
    pdfs = dict()
    for n0 in N0:
        x = list(map(cdf,list(range(50)),np.full(50,n0)))
        pdfs[n0] = pdf(x)
    return pdfs

# virtual patch environment
class PatchEnvironment():
    """
        Virtual foraging environment
        This only does two things:
            1. Return rewards on probabilistic or deterministic reward schedules
            2. Return patch ON state or patch OFF state
            3. Does need to keep track of time for logging purposes?
            3.5. Log output data as this is what is going to match what we observe?
    """
    def __init__(self,rew_system = 'probabilistic',nTimestates = 50,ITI_penalty = 3,timecost = 0):
        self.rew_system = rew_system
        self.nTimestates = nTimestates
        self.ITI_penalty = ITI_penalty
        self.N0 = [.5, .25, .125] # [.5, .25, .125]
        self.pdfs = generate_pdfs(self.N0)
        self.rewsizes = [1, 2, 4]
        self.state = ITI_state
        self.timecost = timecost

    def execute_action(self, action, probe_trial = {}):
        """
            Environment changes state, returns reward based on agent action
            Arguments: action {STAY or LEAVE}
            Returns: new state, reward
        """
        if action == STAY and (self.state["patch"] == ON) and (self.state["t"] < len(self.state["rews"]) - 1):
            self.state["t"] += 1 # increment time on patch
            rew = self.state["rews"][self.state["t"]] - self.timecost # give reward acc to current schedule
        # patch eviction
        elif self.state["patch"] == ON and (self.state["t"] >= len(self.state["rews"]) - 1):
            self.state = ITI_state
            rew = - self.ITI_penalty - self.timecost
        elif action == LEAVE and self.state["patch"] == ON:
            self.state = ITI_state
            rew = - self.ITI_penalty - self.timecost
        elif action == STAY and self.state["patch"] == OFF: # don't change state
            rew = - self.timecost
        else: # action == LEAVE and self.state["patch"] == OFF
            self.new_patch(probe_trial = probe_trial) # this changes the env state
            rew = self.state["rews"][self.state["t"]] - self.timecost # deliver the first reward
        return rew

    def new_patch(self,probe_trial = {}):
        # normal random trial
        if len(probe_trial) == 0:
            curr_rews = np.zeros(self.nTimestates)
            curr_rewsize = rnd.choice(self.rewsizes)
            curr_rewfreq = rnd.choice(self.N0)
            if self.rew_system == 'deterministic':
                curr_rews[[0,4,16]] = curr_rewsize
            if self.rew_system == 'probabilistic':
                curr_rewlocs = np.where(rnd.random(50) - self.pdfs[curr_rewfreq] < 0)[0].tolist()
                curr_rews[curr_rewlocs] = curr_rewsize
            curr_rewindex = self.rewsizes.index(curr_rewsize)
            self.state = {"patch" : ON , "rewsize" : curr_rewsize,"rewindex" : curr_rewindex,
                               "n0" : curr_rewfreq , "rews" : curr_rews,"t" : 0}
        else:
            curr_rews = probe_trial["rews"]
            curr_rewsize = probe_trial["rews"][0]
            curr_rewindex = self.rewsizes.index(curr_rewsize)
            curr_rewfreq = probe_trial["n0"]  # change this later, but for now just
            self.state = {"patch" : ON , "rewsize" : curr_rewsize,"rewindex" : curr_rewindex,
                               "n0" : curr_rewfreq , "rews" : curr_rews,"t" : 0}
