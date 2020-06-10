from numpy import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

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
