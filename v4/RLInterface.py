from numpy import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

STAY = 0
LEAVE = 1

OFF = 0
ON = 1

ITI_state = {"patch" : OFF , "rewindex" : -1}

class RLInterface():
    """
        Bring together an agent and an environment and THROW DOWN
        Methods: step, run_trials, various visualizations
    """
    def __init__(self,agent,environment):
        self.agent = agent
        self.env = environment

    def step(self,probe_trial = {},probe_action = -1):
        old_state = self.env.state.copy()
        old_rew_index = old_state["rewindex"]
        old_rew_int = self.agent.integrate(old_state) # internalize old env state
        old_patch = old_state["patch"]

        if probe_action < 0:
            action = self.agent.select_action(old_rew_index,old_rew_int,self.env.state["patch"]) # agent selects action
        else:
            action = probe_action
        rew = self.env.execute_action(action,probe_trial = probe_trial) # execute agent action into the environment
        new_state = self.env.state.copy()
        new_rew_index = self.env.state["rewindex"]
        new_rew_int = self.agent.integrate(new_state) # internalize new env state
        new_patch = new_state["patch"]

        rpe = self.agent.update(old_rew_index, old_rew_int,old_patch,
                                new_rew_index, new_rew_int,new_patch, action, rew) # update Q
        if self.env.state["patch"] == ON:
            value = self.agent.Q[ON][old_rew_index,STAY,old_rew_int]
        elif self.env.state["patch"] == OFF:
            value = self.agent.Q[OFF][LEAVE]

        return action,rew,rpe,value

    def run_trials(self,nTrials,probe_specs = [],return_ll = False):
        """
            probe_specs is a list of dictionaries with reward vector and n0 values
            ie probe_specs = [{"rews" : [1,1,1,1,1,0,0,0,0,0] , "n0" : .125},
                              {"rews" : [1,1,1,1,1,0,0,0,0,0] , "n0" : .125}]
        """
        actions = []
        # dataframe to end all datastructures
        self.prt_df = pd.DataFrame(columns = ["rewsize","N0","rewsizeN0","PRT"])
        self.mvt_df = pd.DataFrame(columns = ["trial","timepoint","rew","instTrue","avgEst","v_patch"])

        self.prts = {1:[],2:[],4:[]} # Patch residence times divided by reward size
        self.rew_locs = {1:[],2:[],4:[]} # Reward locations for heatmap visualization

        self.rews = [] # Flat vector of reward received for efficiency visualization
        self.rpes = {1:[],2:[],4:[]} # for rpe heatmap visualization
        self.values = {1:[],2:[],4:[]} # for value heatmap visualization
        self.rews_trialed = {1:[],2:[],4:[]} # for heatmap visualization
        self.q_iti = []
        self.q_leave = []

        rewavg_window = 500

        # run trials with no probe trials
        if len(probe_specs) == 0:
            for iTrial in range(nTrials):
                q_patch_list = []
                avg_list = []

                # Start with patch off
                while self.env.state["patch"] == OFF:
                    action,rew,rpe,value = self.step()
                    self.rews.append(rew)
                    if len(self.rews) > rewavg_window:
                        avg_list.append(np.mean(self.rews[-rewavg_window:]))
                    else:
                        avg_list.append(np.mean(self.rews))
                    q_patch_list.append(self.agent.Q[OFF][LEAVE])

                # initialize trial record keeping datastructures
                self.curr_rew = self.env.state["rewsize"]
                self.curr_freq = self.env.state["n0"]
                self.rew_locs[self.curr_rew].append(self.env.state["rews"])
                curr_prt = 0
                curr_rpes = []
                curr_rew_rec = []
                curr_values = []

                while self.env.state["patch"] == ON: # now behave on patch
                    action,rew,rpe,value = self.step()
                    curr_rew_rec.append(rew)
                    actions.append(action)
                    curr_rpes.append(rpe)
                    curr_values.append(value)
                    self.rews.append(rew)
                    q_patch_list.append(value)
                    curr_prt += 1
                    if len(self.rews) > rewavg_window:
                        avg_list.append(np.mean(self.rews[-rewavg_window:]))
                    else:
                        avg_list.append(np.mean(self.rews))

                # record data after leaving
                self.rews_trialed[self.curr_rew].append(curr_rew_rec)

                self.rpes[self.curr_rew].append(curr_rpes)
                self.values[self.curr_rew].append(curr_values)
                self.prts[self.curr_rew].append(curr_prt)

                # dataframe structure
                self.prt_df.at[iTrial] = [self.curr_rew,self.curr_freq,self.curr_rew+self.curr_freq,curr_prt]

                # dataframe for mvt measures
                trial_list = [iTrial for i in range(len(q_patch_list))]
                timepoint_list = [i for i in range(len(q_patch_list))]
                rews = self.rews[-len(q_patch_list):]
                true_inst = self.env.pdfs[self.curr_freq][:len(q_patch_list)]
                true_inst = [x * self.curr_rew - self.env.timecost for x in true_inst]
                curr_mvt_array = np.array([trial_list,timepoint_list,rews,true_inst,avg_list,q_patch_list]).T
                curr_mvt_df = pd.DataFrame(curr_mvt_array,columns = ["trial","timepoint","rew","instTrue","avgEst","v_patch"])
                self.mvt_df = self.mvt_df.append(curr_mvt_df)

                # update dynamic values
                if self.agent.dynamic_beta == True:
                    self.agent.beta = (self.agent.beta_final + (self.agent.beta0 - self.agent.beta_final) *
                                                            np.exp(-1. * iTrial / self.agent.beta_decay))
                if self.agent.dynamic_lr == True:
                    self.agent.lr = (self.agent.lr_final + (self.agent.lr0 - self.agent.lr_final) *
                                                            np.exp(-1. * iTrial / self.agent.lr_decay))
                self.q_iti.append(self.agent.Q[OFF][LEAVE])
                self.q_leave.append(self.agent.Q[ON][0,LEAVE,0])

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
                curr_rpes = []
                curr_rew_rec = []
                curr_values = []

                while self.env.state["patch"] == ON: # now behave on patch
                    action,rew,rpe,value = self.step(probe_trial = probe_specs[iTrial])
                    curr_rew_rec.append(rew)
                    actions.append(action)
                    curr_rpes.append(rpe)
                    curr_values.append(value)
                    self.rews.append(rew)
                    curr_prt += 1

                # record data after leaving
                self.rews_trialed[self.curr_rew].append(curr_rew_rec)
                self.rpes[self.curr_rew].append(curr_rpes)
                self.values[self.curr_rew].append(curr_values)
                self.prts[self.curr_rew].append(curr_prt)
                self.q_iti.append(self.agent.Q[OFF][LEAVE])
                self.q_leave.append(self.agent.Q[ON][0,LEAVE,0])

    def show_qtable(self):
        """
            A visualization method to analyze how the agent is making decisions
        """
        plt.figure()
        plt.subplots_adjust(hspace = 15,wspace = .5)
        if self.agent.model == "Model1":
            plt.subplot(1,2,1)
            plt.title('Patch ON STAY Q table')
            sns.heatmap(self.agent.Q[ON][:,STAY,:10])
        elif self.agent.model == "Model2":
            plt.subplot(1,2,1)
            plt.title('Patch ON STAY Q table')
            sns.heatmap(self.agent.Q[ON][:,STAY,:7])
        elif self.agent.model == "Model3":
            plt.subplot(1,2,1)
            plt.title('Patch ON STAY Q table')
            sns.heatmap(self.agent.Q[ON][:,STAY,145:165])
        sns.set()
        plt.subplot(1,2,2)
        plt.title('Environmental Value Estimation Over Time')
        print("here")
        plt.plot(gaussian_filter(self.q_iti,50),label = "Q[PatchOFF,LEAVE]")
        plt.plot(gaussian_filter(self.q_leave,50),label = "Q[PatchON,LEAVE]")
        plt.legend()
        plt.suptitle('%s Q Table'%(self.agent.model))
        sns.reset_orig()

    def plot_percent_stay(self,decisions):
        """
            Just visualize the proportion of stay decisions agent is making on the patch
        """
        percent = [1 - sum(decisions[0:i])/i for i in range(1,len(decisions))]
        plt.figure()
        plt.title('Percent of STAY choice on patch over time')
        plt.ylim([0,1])
        plt.plot(percent)

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

    def plot_rewrate(self,sd_filter,agent_type,irange = []):
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

    def barcode_beh(self):
        # print(self.prts)
        for patch in self.prts.keys():
            these_prts = self.prts[patch]
            if len(these_prts) > 0:
                max_prt = max(these_prts)
                barcodes = np.zeros((len(these_prts),max_prt))
                for trial in range(len(these_prts)):
                    barcodes[trial,:these_prts[trial]] = 1
                plt.figure()
                sns.heatmap(barcodes)
                plt.title("%i uL Patch Behavior Over Trials" % patch)

    def plot_qiti(self):
        plt.figure()
        plt.plot(self.q_iti,color = [0,0,0])
        plt.title('ITI Leave Q Value over trials')


    def bs_hmap(self,timeRange,trialRange):
        """
            Plot optimistic belief state according to markov model with reward integration data
        """
