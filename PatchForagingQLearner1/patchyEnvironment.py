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
    def __init__(self,rew_system = 'deterministic',nTimestates = 50,ITI_len = 5):
        self.rew_system = rew_system
        self.nTimestates = nTimestates
        self.ITI_len = ITI_len
        self.N0 = [.5, .25, .125]
        self.pdfs = generate_pdfs(self.N0)
        self.rewsizes = [1, 2, 4]
        self.state = [OFF,0] # start with the patch off with no distance run

    def execute_action(self, action, rewrate_probe):
        """
            Environment changes state, returns reward based on agent action
            Arguments: action {STAY or LEAVE}
            Returns: new state, reward
        """
        if action == STAY and self.state[0] == ON:
            self.state[1] += 1 # increment time on patch
            rew = self.curr_rews[self.state[1]] # give reward acc to current schedule
        elif action == LEAVE and self.state[0] == ON: # leave patch
            self.state = [OFF,0]
            rew = 0

        elif action == STAY and self.state[0] == OFF:
            # state stays the same
            rew = 0
        elif action == LEAVE and self.state[0] == OFF:
            self.state[1] += 1
            if self.state[1] == self.ITI_len:
                self.new_patch(rewrate_probe) # this changes the env state
                rew = self.curr_rews[0] # deliver the first reward
            else:
                rew = 0
        return self.state, rew

    def new_patch(self, rewrate_probe):
        self.curr_rews = np.zeros(self.nTimestates)

        curr_rewsize = rnd.choice(self.rewsizes)
        # curr_rewsize = 1 # for this test
        if rewrate_probe == 0:
            curr_rewfreq = rnd.choice(self.N0)
        elif rewrate_probe == 1: # high reward rate zone!
            curr_rewfreq = .5
        elif rewrate_probe == -1: # low reward rate zone!
            curr_rewfreq = .125

        self.curr_freq = curr_rewfreq
        self.curr_rewsize = curr_rewsize

        if self.rew_system == 'deterministic':
            self.curr_rews[[0,4,16]] = curr_rewsize
        if self.rew_system == 'probabilistic':
            curr_rewlocs = np.where(rnd.random(50) - self.pdfs[curr_rewfreq] < 0)[0].tolist()
            self.curr_rews[curr_rewlocs] = curr_rewsize

        self.state = [ON,0]
