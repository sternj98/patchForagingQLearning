import random
from numpy import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter
import progressbar

STAY = 0
LEAVE = 1

OFF = 0
ON = 1

ITI_state = {"patch" : OFF , "rewindex" : -1}

class DeepRLInterface():
    def __init__(self,agent,environment):
        self.agent = agent
        self.env = environment
        self.batch_size = 50
        self.maxRew = 4

    def step(self,probe_trial = {},probe_action = -1):
        env_state = self.env.state.copy()
        int_state = self.agent.integrate(env_state) # internalize old env state

        if probe_action < 0:
            action = self.agent.select_action(int_state) # agent selects action
        else:
            action = probe_action

        rew = self.env.execute_action(action,probe_trial = probe_trial) / self.maxRew # execute agent action into the environment
        new_env_state = self.env.state.copy()
        new_int_state = self.agent.integrate(new_env_state) # internalize new env state

        self.agent.replay_buffer.push(int_state,action,new_int_state,rew) # add our memory to exp buffer

        # can get rpe and value here
        valueAction = self.agent.policy_net.forward(int_state)[action] # value of staying
        EV_new = max(self.agent.policy_net.forward(new_int_state))
        rpe = rew + self.agent.gamma * EV_new - valueAction

        if self.env.state["patch"] == ON:
            valueStay = self.agent.policy_net.forward(int_state)[0]
        elif self.env.state["patch"] == OFF:
            valueStay = self.agent.policy_net.forward(int_state)[1]

        loss = self.agent.update(self.batch_size)
        self.losslist.append(loss)

        return action,rew,rpe,valueStay

    def run_trials(self,nTrials,probe_specs = []):
        counter = 0 # for batch training

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
        # self.q_leave = []
        self.q_iti = []

        rewavg_window = 500
        self.eps = [] # for monitoring eps over time
        self.losslist = []

        widgets = [progressbar.Percentage(), progressbar.Bar()]
        bar = progressbar.ProgressBar(widgets=widgets,maxval=nTrials).start()

        # run trials with no probe trials
        for iTrial in range(nTrials):
            bar.update(iTrial)

            q_patch_list = []
            avg_list = []

            # Start with patch off
            while self.env.state["patch"] == OFF:
                counter += 1
                if len(probe_specs) == 0:
                    action,rew,rpe,value = self.step()
                else:
                    action,rew,rpe,value = self.step(probe_trial = probe_specs[iTrial])
                self.rews.append(rew)
                if len(self.rews) > rewavg_window:
                    avg_list.append(np.mean(self.rews[-rewavg_window:]))
                else:
                    avg_list.append(np.mean(self.rews))

                q_patch_list.append(float(value.detach().numpy()))

            # initialize trial record keeping datastructures
            self.curr_rew = self.env.state["rewsize"]
            self.curr_freq = self.env.state["n0"]
            self.rew_locs[self.curr_rew].append(self.env.state["rews"])
            curr_prt = 0
            curr_rpes = []
            curr_rew_rec = []
            curr_values = []

            while self.env.state["patch"] == ON: # now behave on patch
                counter += 1
                if len(probe_specs) == 0:
                    action,rew,rpe,value = self.step()
                else:
                    action,rew,rpe,value = self.step(probe_trial = probe_specs[iTrial])
                curr_rew_rec.append(rew)
                actions.append(action)
                curr_rpes.append(rpe)
                curr_values.append(float(value.detach().numpy()))
                self.rews.append(rew)
                q_patch_list.append(float(value.detach().numpy()))
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
            if len(q_patch_list) == 51: # dimension issue if we stay the whole trial
                true_inst = true_inst + [0]
            curr_mvt_array = np.array([trial_list,timepoint_list,rews,true_inst,avg_list,q_patch_list]).T
            try:
                curr_mvt_df = pd.DataFrame(curr_mvt_array,columns = ["trial","timepoint","rew","instTrue","avgEst","v_patch"])
            except:
                print([len(l) for l in [trial_list,timepoint_list,rews,true_inst,avg_list,q_patch_list]])
            # curr_mvt_df = pd.DataFrame(curr_mvt_array,columns = ["trial","timepoint","rew","instTrue","avgEst","v_patch"])
            self.mvt_df = self.mvt_df.append(curr_mvt_df)

            self.q_iti.append(float(self.agent.policy_net([0,1,0,0])[LEAVE].detach().numpy()))

            # # update dynamic values
            # if self.agent.dynamic_beta == True:
            #     self.agent.beta = (self.agent.beta_final + (self.agent.beta0 - self.agent.beta_final) *
            #                                             np.exp(-1. * iTrial / self.agent.beta_decay))
            # if self.agent.dynamic_lr == True:
            #     self.agent.lr = (self.agent.lr_final + (self.agent.lr0 - self.agent.lr_final) *
            #                                             np.exp(-1. * iTrial / self.agent.lr_decay))

            # the only unique thing we need to do w/ the deep agent
            if counter % self.agent.target_update == 0: # update the target network
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
            # update agent epsilon
            self.agent.epsilon = (self.agent.eps_end + (self.agent.eps_start - self.agent.eps_end) *
                                                        np.exp(-1. * iTrial / self.agent.eps_decay))
            self.eps.append(self.agent.epsilon)
            self.agent.beta = (self.agent.beta_final + (self.agent.beta0 - self.agent.beta_final) *
                                                        np.exp(-1. * iTrial / self.agent.beta_decay))

    def show_qtable(self):
        """
            A visualization method to analyze how the agent is making decisions
        """
        plt.figure()
        plt.subplot(1,2,1)
        # first plot patch value estimate
        resolution = 500 # resolution of function analysis
        Q_mat = np.zeros((resolution,len(self.env.rewsizes)))
        for i,r in enumerate(np.array(self.env.rewsizes) / max(self.env.rewsizes)):
            states = [[1,r,intValue] for intValue in np.linspace(0,1.,resolution)]
            fn = self.agent.policy_net(torch.FloatTensor(states).float()).detach().numpy()
            Q_mat[:,i] = fn[:,0] - fn[:,1] # relative value
        sns.heatmap(Q_mat,xticklabels = [1,2,4])
        plt.title("Relative Value of Staying acr int values")

        # plot environment value est
        sns.set()
        plt.subplot(1,2,2)
        plt.title('Environmental Value Estimation Over Time')
        plt.plot(gaussian_filter(self.q_iti,50),label = "Q(PatchOFF,LEAVE)")
        plt.legend()
        plt.suptitle('%s Q Function'%(self.agent.model))
        sns.reset_orig()

    # plot loss over time
    def plot_loss(self):
        plt.figure()
        plt.plot(self.losslist)
        plt.title("Loss over training")
