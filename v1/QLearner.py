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

class RLInterface():
    """
        Bring together an agent and an environment and THROW DOWN
        Methods: step, run_trials, various visualizations
    """
    def __init__(self,agent,environment):
        self.agent = agent
        self.env = environment

    def step(self,record_decisions,rewrate_probe = 0):
        old_state = self.env.state # get env state
        internal_old_state = self.agent.internalize_state(old_state) # internalize old env state
        action = self.agent.select_action(internal_old_state) # agent selects action
        new_state,rew = self.env.execute_action(action ,rewrate_probe) # execute agent action into the environment
        if rew > 0:
            if self.agent.omniscient != True:
                self.agent.integrate(rew)
            else:
                self.agent.integrate(env.curr_rewsize,env.curr_freq)
        internal_new_state = self.agent.internalize_state(new_state) # internalize new env state
        rpe = self.agent.update(internal_old_state, internal_new_state, action, rew) # update Q
        if len(internal_old_state) == 3: # right now only look when on patch
            value = self.agent.Q[1][(STAY,)+ internal_old_state[1:]]
        else:
            value = 0
        if record_decisions == True:
            return action,rew,rpe,value

    def run_trials(self,nTrials,epsilon_decay = 0,record_decisions = False,high_rew = [],low_rew = []):
        if record_decisions == True:
            decisions = []
            self.prts = {1:[],2:[],4:[]}
            self.rew_locs = {1:[],2:[],4:[]}
            self.prts_plus = {1:{.125:[],.25:[],.5:[]},
                              2:{.125:[],.25:[],.5:[]},
                              4:{.125:[],.25:[],.5:[]}}
            self.rews = []
            self.rpes = {1:[],2:[],4:[]}
            self.values = {1:[],2:[],4:[]}
            self.rews_trialed = {1:[],2:[],4:[]}

        for iTrial in range(nTrials):
            if len(high_rew) > 0 and iTrial in range(high_rew[0],high_rew[1]):
                ienv_probe = 1
            elif len(low_rew) > 0 and iTrial in range(low_rew[0],low_rew[1]):
                ienv_probe = -1
            else:
                ienv_probe = 0

            while self.env.state[0] == OFF: # start with patch off
                action,rew,rpe,value = self.step(record_decisions,rewrate_probe = ienv_probe)
                self.rews.append(rew)
            # patch on, record parameters
            self.curr_rew = int(self.env.curr_rews[0])
            self.curr_freq = self.env.curr_freq
            self.rew_locs[self.curr_rew].append(self.env.curr_rews)

            self.curr_prt = 0
            curr_rpes = []
            curr_rew_rec = []
            curr_values = []
            while self.env.state[0] == ON: # now allow for behavior on the patch
                if record_decisions == True:
                    action,rew,rpe,value = self.step(record_decisions,rewrate_probe = ienv_probe)
                    self.curr_prt += 1
                    curr_rew_rec.append(rew)
                    decisions.append(action)
                    curr_rpes.append(rpe)
                    curr_values.append(value)
                    self.rews.append(rew)
                else:
                    self.step(record_decisions)

            self.rews_trialed[self.curr_rew].append(curr_rew_rec)
            self.rpes[self.curr_rew].append(curr_rpes)
            self.values[self.curr_rew].append(curr_values)
            self.prts[self.curr_rew].append(self.curr_prt)
            self.prts_plus[self.curr_rew][self.curr_freq].append(self.curr_prt)

            self.agent.reset_integration()
            self.agent.epsilon = self.agent.epsilon0 * np.e**(-iTrial/epsilon_decay) + self.agent.baseline_epsilon

        if record_decisions == True:
            return decisions

    def show_qtable(self):
        """
            A visualization method to analyze how the agent is making decisions
        """
        plt.figure(figsize = (10,3))
        plt.subplots_adjust(hspace = 15,wspace = .5)
        plt.suptitle('Agent Q Table',x=.5,y=1.05)
        plt.subplot(1,3,1)
        plt.title('Patch ON STAY Q table')
        sns.heatmap(self.agent.Q[1][0,:30,0:12])
        plt.subplot(1,3,2)
        plt.title('Patch ON LEAVE Q table')
        sns.heatmap(self.agent.Q[1][1,:30,0:12])
        plt.subplot(1,3,3)
        plt.title('Patch OFF Q table')
        sns.heatmap(self.agent.Q[0])
        plt.show()

    def plot_percent_stay(self,decisions):
        """
            Just visualize the proportion of stay decisions agent is making on the patch
        """
        percent = [1 - sum(decisions[0:i])/i for i in range(1,len(decisions))]
        plt.figure()
        plt.title('Percent of STAY choice on patch over time')
        plt.ylim([0,1])
        plt.plot(percent)

    def plot_prts(self,resolution):
        """
            Visualize smoothed PRTs over learning, separated by patch type
            Use this to determine around where behavior stabilizes
        """
        plt.figure()
        for patch in self.prts.keys():
            coords = list(range(0,len(self.prts[patch])-resolution,resolution))
            smoothened = [np.mean(self.prts[patch][coords[i]:coords[i+1]]) for i in range(len(coords)-1)]
            plt.plot(smoothened,label = str(str(patch) + ' uL'))
        plt.legend()
        plt.ylabel('Avg Patch Residence Time')
        plt.xlabel('Time over training')
        plt.title('Patch-Separated Evolution of PRTs over Training')

    def plot_rewrate(self,resolution,agent_type,irange = []):
        """
            Visualize smoothed rewrate over course of learning
            Use this to determine around where behavior stabilizes and how efficient the algorithm is
        """
        plt.figure()
        # print(irange)
        # if len(irange) == 0:
        #     coords = list(range(0,len(self.rews)-resolution,resolution))
        # else:
        #     coords = list(range(irange[0],irange[1],resolution))
        #     print(coords)
        smoothened = [np.mean(self.rews[coords[i]:coords[i+1]]) for i in range(len(coords)-1)]
        plt.plot(smoothened)
        plt.ylabel('Avg Rew/sec')
        plt.ylim([.3,.6])
        plt.xlabel('Time over training')
        plt.title('Rew/sec over Training for ' + agent_type + ' Agent')
        if len(irange) == 0:
            print("Mean:",np.mean(self.rews))
            print("Std:",np.std(self.rews))
        else:
            print("Mean:",np.mean(self.rews[irange[0] : irange[1]]))
            print("Std:",np.std(self.rews[irange[0] : irange[1]]))

    def prt_bars(self,start):
        """
            Visualize proportion of stay decisions agent makes on patch, separated by patch type
            Input start parameter, where we start analysis based on convergence after plot_prts analysis
        """
        data = []
        lengths = []
        for patch in self.prts.keys():
            data.append(self.prts[patch])
            lengths.append(len(np.array(self.prts[patch])))
        min_len = min(lengths)
        array_prts = np.array([data[i][start:min_len] for i in range(len(data))]).T
        self.patch_df = pd.DataFrame(array_prts,columns = ['Sm','Md','Lg'])

        plt.figure()
        ax = sns.barplot(data = self.patch_df,palette = [(0,0,0),(.5,1,1),(0,0,1)],edgecolor=".2")
        plt.xlabel('Rew Size (uL)')
        plt.ylabel('Mean PRT (sec)')
        plt.title('PRT by Reward Size')
        plt.show()

    def prt_hist(self,start):
        """
            Input start parameter, where we start analysis based on convergence after plot_prts analysis
            Basically a more detailed visualization of the prt_bars
        """
        data = []
        lengths = []
        for patch in self.prts.keys():
            data.append(self.prts[patch])
            lengths.append(len(np.array(self.prts[patch])))
        min_len = min(lengths)
        array_prts = np.array([data[i][start:min_len] for i in range(len(data))]).T
        array_prts = array_prts + rnd.normal(0,.3,array_prts.shape)
        self.patch_df = pd.DataFrame(array_prts,columns = ['Sm','Md','Lg'])

        plt.figure(figsize = [5,15])
        plt.subplot(3,1,1)
        plt.title('1 uL Rewsize')
        sns.distplot(self.patch_df["Sm"], color=(0,0,0), hist=False, rug=False)
        plt.subplot(3,1,2)
        plt.title('2 uL Rewsize')
        sns.distplot(self.patch_df["Md"], color=(.5,1,1), hist=False, rug=False)
        plt.subplot(3,1,3)
        plt.title('4 uL Rewsize')
        sns.distplot(self.patch_df["Lg"], color = (0,0,1),hist=False, rug=False)

        # plt.xlabel('Rew Size (uL)')
        plt.suptitle('PRT Distribution by Reward Size')
        plt.show()

    def prt_plus_bars(self,start):
        """
            Visualize mean PRT separated by reward size and frequency
            Input start parameter, where we start analysis based on convergence after plot_prts analysis
        """
        data = []
        lengths = []
        for rew_size in self.prts_plus.keys():
            for rew_freq in self.prts_plus[rew_size].keys():
                data.append(self.prts_plus[rew_size][rew_freq])
                lengths.append(len(self.prts_plus[rew_size][rew_freq]))
        min_len = min(lengths)
        array_prts = np.array([data[i][start:min_len] for i in range(len(data))]).T
        self.patch_plus_df = pd.DataFrame(array_prts,columns = ['1uL Lo','1uL Md','1uL Hi','2uL Lo','2uL Md','2uL Hi','4uL Lo','4uL Md','4uL Hi'])

        colors = [(.5,.5,.5),(.3,.3,.3),(0,0,0),(.9,1,1),(.7,1,1),(.5,1,1),(.5,.5,1),(.3,.3,1),(0,0,1)]
        plt.figure()
        ax = sns.barplot(data = self.patch_plus_df,palette = colors,edgecolor=".2")
        plt.xlabel('Rew Size (uL)')
        plt.ylabel('Mean PRT (sec)')
        plt.title('PRT by Reward Size and Frequency')
        plt.show()

    def mk_timecourse(self):
        """
            Convert PRTs into 'trial timecourses', binary vectors 0 if we have left patch, 1 if we are still on
        """
        data = self.prts
        num_timesteps = 40

        self.timecourses = dict()
        for patch in data.keys(): # could make this a double list comp but i still hold some mercy in my heart
#             for prt in data[patch]:
#                 print(prt)
            self.timecourses[patch] = np.array([list(np.ones(prt)) + list(np.zeros(num_timesteps-prt)) for prt in data[patch]])

    def plot_survival(self):
        """
            Plot timecourses in terms of survival curves separated by patch
        """
        plt.figure()
        colors = {1:(0,0,0),2:(.5,1,1),4:(0,0,1)}
        for patch in self.timecourses.keys():
            survival = np.sum(self.timecourses[patch],axis = 0)/self.timecourses[patch].shape[0]
            plt.plot(survival[:12],label = str(str(patch) + ' uL'),color = colors[patch])
        plt.legend()
        plt.title('Patch Survival Curve')
        plt.xlabel('Time on Patch (seconds)')
        plt.ylabel('% Survival')
        plt.show()

    def percent_hmap(self,timeRange,trialRange):
        """
            Visualize percent stay over time on patch separated by patch type

            Clean this up
        """
        counter = 1

        plt.figure(figsize = [5,15])
        for patch in [1,2,4]:
            rews = np.array(self.rew_locs[patch])/patch
            cumulative_rews = rews.copy()
            for iTime in range(timeRange[1]):
                cumulative_rews[:,iTime] = np.sum(rews[:,:iTime+1],axis = 1)

            max_rew = int(np.max(cumulative_rews[trialRange[0]:trialRange[1],timeRange[0]:timeRange[1]]))

            hmap_num = np.zeros((max_rew,timeRange[1]))
            hmap_denom = np.zeros((max_rew,timeRange[1]))
            for trial in range(trialRange[0],trialRange[1]):
                for time in range(timeRange[0],timeRange[1]):
                    cumulative_rew = int(cumulative_rews[trial,time])
                    hmap_num[cumulative_rew-1,time] += self.timecourses[patch][trial,time]
                    hmap_denom[cumulative_rew-1,time] += 1
            # display(hmap_denom)
            hmap = np.divide(hmap_num,hmap_denom,where = hmap_denom>0)
            hmap[np.where(hmap > 1)[0]] = 0
            plt.subplot(3,1,counter)
            plt.title(str(str(patch) + 'uL Rew Size'))
            ax = sns.heatmap(hmap)
            ax.invert_yaxis()
            plt.xlabel('Time on patch (sec)')
            plt.ylabel('Rewards Received')
            counter += 1
        plt.suptitle('Heatmap of patch stay percentage')
        plt.show()

    def rpe_hmap(self,timeRange,trialRange):
        """
            Visualize history-dependent RPE over time separated by patch type
        """
        counter = 1
        plt.figure(figsize = [5,15])
        for patch in [1,2,4]:
            rews = np.array(self.rew_locs[patch])/patch
            cumulative_rews = rews.copy()
            for iTime in range(timeRange[1]):
                cumulative_rews[:,iTime] = np.sum(rews[:,:iTime+1],axis = 1)

            max_rew = int(np.max(cumulative_rews[trialRange[0]:trialRange[1],timeRange[0]:timeRange[1]]))

            hmap_num = np.zeros((max_rew,timeRange[1]))
            hmap_denom = np.zeros((max_rew,timeRange[1]))
            for trial in range(trialRange[0],trialRange[1]):
                for time in range(min( len(self.rpes[patch][trial])-1,timeRange[1])):
                    cumulative_rew = int(cumulative_rews[trial,time])

                    if self.rews_trialed[patch][trial][time] > 0:
                        hmap_num[cumulative_rew-1,time] += self.rpes[patch][trial][time]
                        hmap_denom[cumulative_rew-1,time] += 1
            # display(hmap_denom)
            hmap = np.divide(hmap_num,hmap_denom,where = hmap_denom>0)
            plt.subplot(3,1,counter)
            plt.title(str(str(patch) + 'uL Rew Size'))
            ax = sns.heatmap(hmap)
            ax.invert_yaxis()
            plt.xlabel('Time on patch (sec)')
            plt.ylabel('Rewards Received')
            counter += 1
        plt.suptitle('RPE Heatmap')
        plt.show()

    def value_hmap(self,timeRange,trialRange):
        """
            Visualize history-dependent value representation over time sep by patch type
        """
        counter = 1
        plt.figure(figsize = [5,15])
        for patch in [1,2,4]:
            rews = np.array(self.rew_locs[patch])/patch
            cumulative_rews = rews.copy()
            for iTime in range(timeRange[1]):
                cumulative_rews[:,iTime] = np.sum(rews[:,:iTime+1],axis = 1)

            max_rew = int(np.max(cumulative_rews[trialRange[0]:trialRange[1],timeRange[0]:timeRange[1]]))

            hmap_num = np.zeros((max_rew,timeRange[1]))
            hmap_denom = np.zeros((max_rew,timeRange[1]))
            for trial in range(trialRange[0],trialRange[1]):
                for time in range(min( len(self.values[patch][trial])-1,timeRange[1])):
                    cumulative_rew = int(cumulative_rews[trial,time])

                    if self.rews_trialed[patch][trial][time] > 0:
                        hmap_num[cumulative_rew-1,time] += self.values[patch][trial][time]
                        # if self.values[patch][trial][time] > 4:
                        #     print(self.values[patch][trial][time])
                        hmap_denom[cumulative_rew-1,time] += 1
            # display(hmap_denom)
            hmap = np.divide(hmap_num,hmap_denom,where = hmap_denom>0)
            plt.subplot(3,1,counter)
            plt.title(str(str(patch) + 'uL Rew Size'))
            ax = sns.heatmap(hmap)
            ax.invert_yaxis()
            plt.xlabel('Time on patch (sec)')
            plt.ylabel('Rewards Received')
            counter += 1
        plt.suptitle('Value Heatmap')
        plt.show()

    def bs_hmap(self,timeRange,trialRange):
        """
            Plot optimistic belief state according to markov model with reward integration data
        """
