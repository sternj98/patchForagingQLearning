from numpy import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

# same deal as RLInterface; bring patchyEnvironment together with MVT agent
# log same datastructures here for easy comparison with RL behavior

STAY = 0
LEAVE = 1

OFF = 0
ON = 1

class MVTInterface():
    """
        Bring together MVT agent with patchy env and THROW DOWN
        Methods: step, run_trials, visualizations
    """
    def __init__(self,agent,environment):
        self.agent = agent
        self.env = environment

    def step(self,probe_trial = {},probe_action = -1):
        old_state = self.env.state.copy()
        old_patch = old_state["patch"]
        if old_patch == ON:
            if probe_action < 0:
                action = self.agent.select_action() # agent selects action
            else:
                action = probe_action
        else:
            action = LEAVE # find new patch

        # update either way
        rew = self.env.execute_action(action,probe_trial = probe_trial) # execute agent action into the environment
        new_state = self.env.state.copy()

        self.agent.update(rew) # update Q

        return action,rew

    def run_trials(self,nTrials,probe_specs = [],return_ll = False):
        """
            probe_specs is a list of dictionaries with reward vector and n0 values
            ie probe_specs = [{"rews" : [1,1,1,1,1,0,0,0,0,0] , "n0" : .125},
                              {"rews" : [1,1,1,1,1,0,0,0,0,0] , "n0" : .125}]
        """
        actions = []
        # dataframe to end all datastructures
        self.prt_df = pd.DataFrame(columns = ["rewsize","N0","rewsizeN0","PRT"])
        self.mvt_df = pd.DataFrame(columns = ["trial","timepoint","rew","instTrue","avgEst","instEst"])

        self.prts = {1:[],2:[],4:[]} # Patch residence times divided by reward size
        self.rew_locs = {1:[],2:[],4:[]} # Reward locations for heatmap visualization
        self.rews_trialed = {1:[],2:[],4:[]} # for heatmap visualization

        self.rews = [] # Flat vector of reward received for efficiency visualization

        # run trials with no probe trials
        if len(probe_specs) == 0:
            for iTrial in range(nTrials):
                inst_list = []
                avg_list = []
                # Start with patch off
                while self.env.state["patch"] == OFF:
                    action,rew = self.step()
                    self.rews.append(rew)
                    inst_list.append(self.agent.inst)
                    avg_list.append(self.agent.avg)

                # initialize trial record keeping datastructures
                self.curr_rew = self.env.state["rewsize"]
                self.curr_freq = self.env.state["n0"]
                self.rew_locs[self.curr_rew].append(self.env.state["rews"])
                curr_prt = 0
                curr_rew_rec = []

                while self.env.state["patch"] == ON: # now behave on patch
                    action,rew = self.step()
                    curr_rew_rec.append(rew)
                    actions.append(action)
                    self.rews.append(rew)
                    inst_list.append(self.agent.inst)
                    avg_list.append(self.agent.avg)
                    curr_prt += 1

                # record data after leaving
                self.rews_trialed[self.curr_rew].append(curr_rew_rec)
                self.prts[self.curr_rew].append(curr_prt)

                # dataframe structure for prt
                self.prt_df.at[iTrial] = [self.curr_rew,self.curr_freq,self.curr_rew+self.curr_freq,curr_prt]
                # dataframe for mvt measures
                trial_list = [iTrial for i in range(len(inst_list))]
                timepoint_list = [i for i in range(len(inst_list))]
                rews = self.rews[-len(inst_list):]
                # true_inst = [-self.env.ITI_penalty] + self.env.pdfs[self.curr_freq][:len(inst_list)-1]
                true_inst = self.env.pdfs[self.curr_freq][:len(inst_list)]
                true_inst = [x * self.curr_rew - self.env.timecost for x in true_inst]
                curr_mvt_array = np.array([trial_list,timepoint_list,rews,true_inst,avg_list,inst_list]).T
                curr_mvt_df = pd.DataFrame(curr_mvt_array,columns = ["trial","timepoint","rew","instTrue","avgEst","instEst"])
                self.mvt_df = self.mvt_df.append(curr_mvt_df)

        # behave under ste trial conditions
        elif len(probe_specs) > 0:
            for iTrial in range(len(probe_specs)):
                # Start with patch off
                while self.env.state["patch"] == OFF:
                    action,rew,rpe,value = self.step(probe_trial = probe_specs[iTrial])
                    self.rews.append(rew)

                # initialize trial record keeping datastructures
                self.curr_rew = self.env.state["rewsize"]
                self.curr_freq = self.env.state["n0"]
                self.rew_locs[self.curr_rew].append(self.env.state["rews"])
                curr_prt = 0
                curr_rew_rec = []

                # this might take some time... potentially add option to turn off
                while self.env.state["patch"] == ON: # now behave on patch
                    action,rew = self.step(probe_trial = probe_specs[iTrial])
                    curr_rew_rec.append(rew)
                    actions.append(action)
                    self.rews.append(rew)
                    curr_prt += 1

                # record data after leaving
                self.rews_trialed[self.curr_rew].append(curr_rew_rec)
                self.prts[self.curr_rew].append(curr_prt)

                # dataframe structure
                self.prt_df.at[iTrial] = [self.curr_rew,self.curr_freq,self.curr_rew+self.curr_freq,curr_prt]
                # dataframe for mvt measures
                trial_list = [iTrial for i in range(len(inst_list))]
                timepoint_list = [i for i in range(len(inst_list))]
                rews = self.rews[-len(inst_list):]
                # true_inst = [-self.env.ITI_penalty] + self.env.pdfs[self.curr_freq][:len(inst_list)-1]
                true_inst = self.env.pdfs[self.curr_freq][:len(inst_list)]
                true_inst = [x * self.curr_rew - self.env.timecost for x in true_inst]
                curr_mvt_array = np.array([trial_list,timepoint_list,rews,true_inst,avg_list,inst_list]).T
                curr_mvt_df = pd.DataFrame(curr_mvt_array,columns = ["trial","timepoint","rew","instTrue","avgEst","instEst"])
                self.mvt_df = self.mvt_df.append(curr_mvt_df)

    def plot_prts(self,sd_filter):
        """
            Visualize smoothed PRTs over learning, separated by patch type
            Use this to determine around where behavior stabilizes
        """
        plt.figure()
        for patch in self.prts.keys():
            prts = self.prt_df[self.prt_df["rewsize"] == patch]["PRT"].astype('float64')
            smooth_prts = gaussian_filter(prts,sd_filter)
            # print(smooth_prts)
            # smooth_prts = gaussian_filter(self.prts[patch],sd_filter)
            plt.plot(smooth_prts,label = str(str(patch) + ' uL'))
        plt.legend()
        plt.ylabel('Avg Patch Residence Time')
        plt.xlabel('Time over training')
        plt.title('Patch-Separated Evolution of PRTs over Training')

    def plot_rewrate(self,sd_filter,irange = []):
        """
            Visualize smoothed rewrate over course of learning
            Use this to determine around where behavior stabilizes and how efficient the algorithm is
        """
        plt.figure()
        if len(irange) == 2:
            smooth_rews = gaussian_filter(self.rews[irange[0],irange[1]],sd_filter)
        else:
            smooth_rews = gaussian_filter(self.rews,sd_filter)
        plt.plot(smooth_rews)
        plt.ylabel('Avg Rew/sec')
        plt.ylim([0,.6])
        plt.xlabel('Time over training')
        plt.title('Rew/sec over Training for ' + 'MVT Agent')
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
        plt.figure()
        sns.barplot(x = "rewsize",y = "PRT",data = self.prt_df[start:],palette = [(0,0,0),(.5,1,1),(0,0,1)],edgecolor=".2")
        plt.xlabel('Rew Size (uL)')
        plt.ylabel('Mean PRT (sec)')
        plt.title('PRT by Reward Size')

    def prt_hist(self,start):
        """
            Input start parameter, where we start analysis based on convergence after plot_prts analysis
            Basically a more detailed visualization of the prt_bars
        """
        plt.figure()
        sns.violinplot(x = "rewsize",y = "PRT",data = self.prt_df[start:].astype('float64'),palette = [(0,0,0),(.5,1,1),(0,0,1)])
        plt.xlabel('Reward Size')
        plt.ylabel('PRT')
        plt.title('PRT distribution by reward size')

    def prt_plus_bars(self,start):
        """
            Visualize mean PRT separated by reward size and frequency
            Input start parameter, where we start analysis based on convergence after plot_prts analysis
        """
        sizeN0 = ['1uL Lo','1uL Md','1uL Hi','2uL Lo','2uL Md','2uL Hi','4uL Lo','4uL Md','4uL Hi']
        colors = [(.5,.5,.5),(.3,.3,.3),(0,0,0),(.9,1,1),(.7,1,1),(.5,1,1),(.5,.5,1),(.3,.3,1),(0,0,1)]
        plt.figure()
        sns.barplot(x = "rewsizeN0",y = "PRT",data = self.prt_df[start:],palette = colors,edgecolor=".2")
        plt.xticks(range(len(sizeN0)),sizeN0)
        plt.xlabel('Rew Size (uL)')
        plt.ylabel('Mean PRT (sec)')
        plt.title('PRT by Reward Size and Frequency')

    def mk_timecourse(self):
        """
            Convert PRTs into 'trial timecourses', binary vectors 0 if we have left patch, 1 if we are still on
        """
        data = self.prts
        num_timesteps = 51

        self.timecourses = dict()
        for patch in data.keys(): # could make this a double list comp but i still hold some mercy in my heart
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

    def mvt_trialPlot(self):
        """
            Visualization of behavior compared to MVT optimality
            Based on figure 1c of Davidson, Hady
        """
