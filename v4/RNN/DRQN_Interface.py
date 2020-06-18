import progressbar

# use DRQN to estimate value if on patch
class DeepRQInterface():
    def __init__(self,agent,environment):
        self.agent = agent
        self.env = environment
        self.rewlist = []
        self.batch_size = 50 # sample 50 experiences when we update

    def stepOFF(self,states): # same process as above
        action = self.agent.select_action(states) # agent selects action
        rew = self.env.execute_action(action) # execute agent action into the environment


        # split all of this shit up by whether new patch is on or off
        # new state is just the same;
        new_state = self.env.state["rew"][self.env.state["t"]]

        if self.env.state["patch"] == ON:
            valueStay = self.agent.policy_net.forward([])[0]
        elif self.env.state["patch"] == OFF:
            valueStay = self.agent.Q_ITI[LEAVE]

        # can get rpe and value here
        loss = self.agent.update(self.batch_size)
        self.losslist.append(loss) # append loss to assess performance over time
        valueAction = self.agent.policy_net.forward(states + [new_state])[action] # value of staying
        EV_new = max(self.agent.policy_net.forward(new_int_state))
        rpe = rew + self.agent.gamma * EV_new - valueAction

        return state,action,rpe,valueStay

    def stepON(self,states): # same process as above
        action = self.agent.select_action(states) # agent selects action
        rew = self.env.execute_action(action) # execute agent action into the environment
        new_state = self.env.state["rew"][self.env.state["t"]]

        loss = self.agent.update(self.batch_size)
        self.losslist.append(loss) # append loss to assess performance over time
        if self.env.state["patch"] == ON:
            valueStay = self.agent.policy_net.forward(int_state)[0]
        elif self.env.state["patch"] == OFF:
            valueStay = self.agent.policy_net.forward(int_state)[1]

        # can get rpe and value here
        valueAction = self.agent.policy_net.forward(states + [new_state])[action] # value of staying
        EV_new = max(self.agent.policy_net.forward(new_int_state))
        rpe = rew + self.agent.gamma * EV_new - valueAction

        return state,action,rpe,valueStay

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
                action,rew,rpe,value = self.step()
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

            states = []

            while self.env.state["patch"] == ON: # now behave on patch
                states.append(env.state["rews"][curr_prt])
                action,rew,rpe,value = self.step(states)
                curr_rew_rec.append(rew)
                actions.append(action)
                curr_rpes.append(rpe)
                curr_values.append(float(value.detach().numpy()))
                self.rews.append(rew)
                q_patch_list.append(float(value.detach().numpy()))

                if len(self.rews) > rewavg_window:
                    avg_list.append(np.mean(self.rews[-rewavg_window:]))
                else:
                    avg_list.append(np.mean(self.rews))

                counter += 1
                curr_prt += 1

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

            # the only unique thing we need to do w/ the deep agent
            if counter % self.agent.target_update == 0: # update the target network
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
            # update agent epsilon
            self.agent.epsilon = (self.agent.eps_end + (self.agent.eps_start - self.agent.eps_end) *
                                                        np.exp(-1. * iTrial / self.agent.eps_decay))
            self.eps.append(self.agent.epsilon)
            self.agent.beta = (self.agent.beta_final + (self.agent.beta0 - self.agent.beta_final) *
                                                        np.exp(-1. * iTrial / self.agent.beta_decay))



            while not self.env.episode_complete: # while the game is not over, keep taking actions
                state, action, new_state, rew = self.step(states)
                total_rew += rew
                states.append(state)
                actions.append(action)
                rews.append(rew)
                new_states.append(new_state)
                counter += 1

            self.agent.replay_buffer.push(states,actions,new_states,rews)

            if i % assessmentInterval == 0:
                plt.plot(self.env.population.nInf,color = [i/nTrials,0,1 - i/nTrials])
            self.rewlist.append(total_rew)

            if counter % self.agent.target_update == 0: # update the target network
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
            # update agent epsilon
            self.agent.epsilon = (self.agent.eps_end + (self.agent.eps_start - self.agent.eps_end) *
                                                        np.exp(-1. * counter / self.agent.eps_decay))
            self.eps.append(self.agent.epsilon)
