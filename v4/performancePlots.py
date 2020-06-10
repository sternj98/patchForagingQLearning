import random
from numpy import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress as linregress
from scipy.ndimage.filters import gaussian_filter

def percent_hmap(interface,timeRange,trialRange):
    """
        Visualize percent stay over time on patch separated by patch type

        Clean this up
    """
    counter = 1

    plt.figure(figsize = [5,15])
    for patch in [1,2,4]:
        rews = np.array(interface.rew_locs[patch])/patch
        cumulative_rews = rews.copy()
        for iTime in range(timeRange[1]):
            cumulative_rews[:,iTime] = np.sum(rews[:,:iTime+1],axis = 1)

        max_rew = int(np.max(cumulative_rews[trialRange[0]:trialRange[1],timeRange[0]:timeRange[1]]))

        hmap_num = np.zeros((max_rew,timeRange[1]))
        hmap_denom = np.zeros((max_rew,timeRange[1]))
        for trial in range(trialRange[0],trialRange[1]):
            for time in range(timeRange[0],timeRange[1]):
                cumulative_rew = int(cumulative_rews[trial,time])
                hmap_num[cumulative_rew-1,time] += interface.timecourses[patch][trial,time]
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

def plot_percent_stay(self,decisions):
    """
        Just visualize the proportion of stay decisions agent is making on the patch
    """
    percent = [1 - sum(decisions[0:i])/i for i in range(1,len(decisions))]
    plt.figure()
    plt.title('Percent of STAY choice on patch over time')
    plt.ylim([0,1])
    plt.plot(percent)

def barcode_beh(interface):
    # print(interface.prts)
    for patch in interface.prts.keys():
        these_prts = interface.prts[patch]
        if len(these_prts) > 0:
            max_prt = max(these_prts)
            barcodes = np.zeros((len(these_prts),max_prt))
            for trial in range(len(these_prts)):
                barcodes[trial,:these_prts[trial]] = 1
            plt.figure()
            sns.heatmap(barcodes)
            plt.title("%i uL Patch Behavior Over Trials" % patch)

def plot_prts(interface,sd_filter):
    """
        Visualize smoothed PRTs over learning, separated by patch type
        Use this to determine around where behavior stabilizes
    """
    plt.figure()
    for patch in interface.prts.keys():
        prts = interface.prt_df[interface.prt_df["rewsize"] == patch]["PRT"].astype('float64')
        smooth_prts = gaussian_filter(prts,sd_filter)
        # print(smooth_prts)
        # smooth_prts = gaussian_filter(interface.prts[patch],sd_filter)
        plt.plot(smooth_prts,label = str(str(patch) + ' uL'))
    plt.legend()
    plt.ylabel('Avg Patch Residence Time')
    plt.xlabel('Time over training')
    plt.title('Patch-Separated Evolution of PRTs over Training')

def plot_rewrate(interface,sd_filter,irange = []):
    """
        Visualize smoothed rewrate over course of learning
        Use this to determine around where behavior stabilizes and how efficient the algorithm is
    """
    plt.figure()
    if len(irange) == 2:
        smooth_rews = gaussian_filter(interface.rews[irange[0],irange[1]],sd_filter)
    else:
        smooth_rews = gaussian_filter(interface.rews,sd_filter)
    plt.plot(smooth_rews)
    plt.ylabel('Avg Rew/sec')
    plt.ylim([0,.6])
    plt.xlabel('Time over training')
    plt.title('Rew/sec over Training for ' + 'MVT Agent')
    if len(irange) == 0:
        print("Mean:",np.mean(interface.rews))
        print("Std:",np.std(interface.rews))
    else:
        print("Mean:",np.mean(interface.rews[irange[0] : irange[1]]))
        print("Std:",np.std(interface.rews[irange[0] : irange[1]]))

def prt_bars(interface,start):
    """
        Visualize proportion of stay decisions agent makes on patch, separated by patch type
        Input start parameter, where we start analysis based on convergence after plot_prts analysis
    """
    plt.figure()
    sns.barplot(x = "rewsize",y = "PRT",data = interface.prt_df[start:],palette = [(0,0,0),(.5,1,1),(0,0,1)],edgecolor=".2")
    plt.xlabel('Rew Size (uL)')
    plt.ylabel('Mean PRT (sec)')
    plt.title('PRT by Reward Size')

def prt_hist(interface,start):
    """
        Input start parameter, where we start analysis based on convergence after plot_prts analysis
        Basically a more detailed visualization of the prt_bars
    """
    plt.figure()
    sns.violinplot(x = "rewsize",y = "PRT",data = interface.prt_df[start:].astype('float64'),palette = [(0,0,0),(.5,1,1),(0,0,1)])
    plt.xlabel('Reward Size')
    plt.ylabel('PRT')
    plt.title('PRT distribution by reward size')

def prt_plus_bars(interface,start):
    """
        Visualize mean PRT separated by reward size and frequency
        Input start parameter, where we start analysis based on convergence after plot_prts analysis
    """
    sizeN0 = ['1uL Lo','1uL Md','1uL Hi','2uL Lo','2uL Md','2uL Hi','4uL Lo','4uL Md','4uL Hi']
    colors = [(.5,.5,.5),(.3,.3,.3),(0,0,0),(.9,1,1),(.7,1,1),(.5,1,1),(.5,.5,1),(.3,.3,1),(0,0,1)]
    plt.figure()
    sns.barplot(x = "rewsizeN0",y = "PRT",data = interface.prt_df[start:],palette = colors,edgecolor=".2")
    plt.xticks(range(len(sizeN0)),sizeN0)
    plt.xlabel('Rew Size (uL)')
    plt.ylabel('Mean PRT (sec)')
    plt.title('PRT by Reward Size and Frequency')

def mk_timecourse(interface):
    """
        Convert PRTs into 'trial timecourses', binary vectors 0 if we have left patch, 1 if we are still on
    """
    data = interface.prts
    num_timesteps = 51

    interface.timecourses = dict()
    for patch in data.keys(): # could make this a double list comp but i still hold some mercy in my heart
        interface.timecourses[patch] = np.array([list(np.ones(prt)) + list(np.zeros(num_timesteps-prt)) for prt in data[patch]])

def plot_survival(interface):
    """
        Plot timecourses in terms of survival curves separated by patch
    """
    plt.figure()
    colors = {1:(0,0,0),2:(.5,1,1),4:(0,0,1)}
    for patch in interface.timecourses.keys():
        survival = np.sum(interface.timecourses[patch],axis = 0)/interface.timecourses[patch].shape[0]
        plt.plot(survival[:12],label = str(str(patch) + ' uL'),color = colors[patch])
    plt.legend()
    plt.title('Patch Survival Curve')
    plt.xlabel('Time on Patch (seconds)')
    plt.ylabel('% Survival')

def plot_avgrew(rew_df):
    plt.figure()
    sns.set()
    # need to conver this to do lineplot
    rew_df[["timepoint","reward"]] = rew_df[["timepoint","reward"]].astype("float64")
    sns.lineplot(x="timepoint", y="reward", hue = "agent", data=rew_df)
    plt.ylabel('Average Reward/Sec')
    # plt.ylim([-.5,1])
    plt.xlabel('Timestep in training')
    plt.title('Average Reward/Sec Over 1000 Trials')

def plot_prt_slopes(prt_df,rollover = 2):
    plt.figure()
    slopes_df = pd.DataFrame(columns = ["agent","trial","rewsize","slope","std_err"])

    agents = prt_df["agent"].unique()
    rewsizeN0_groups = np.split(np.sort(prt_df["rewsizeN0"].unique()),3)
    rewsizes = np.sort(prt_df["rewsize"].unique())
    trials = np.sort(prt_df["trial"].unique())

    print(agents)

    # iterate over agents
    for agent in agents:
        slope_list = []
        std_err_list = []
        trial_list = []
        rewsize_list = []
        agent_df = prt_df[prt_df["agent"] == agent]
        for rewsize,rewsizeN0_group in zip(rewsizes,rewsizeN0_groups):
            for trial in trials:
                rewsizeN0_prt = agent_df[(agent_df["trial"] == trial) & (np.isin(agent_df["rewsizeN0"], rewsizeN0_group))][["rewsizeN0","PRT"]].to_numpy().astype("float64")
                # linear regression
                slope, intercept, r_value, p_value, std_err = linregress(rewsizeN0_prt[:,0],rewsizeN0_prt[:,1])
                slope_list.append(slope)
                std_err_list.append(std_err)
                trial_list.append(trial)
                rewsize_list.append(rewsize)
        agent_list = [agent for i in range(len(slope_list))]

        slopes_array = np.array((agent_list,trial_list,rewsize_list,slope_list,std_err_list)).T
        slopes_df = slopes_df.append(pd.DataFrame(slopes_array.copy(),columns = ["agent","trial","rewsize","slope","std_err"]),sort = True)

    slopes_df[["slope","rewsize","trial","std_err"]] = slopes_df[["slope","rewsize","trial","std_err"]].astype("float64")
    slopes_df = slopes_df.sort_values("trial")
    colors = [(0.5430834294502115, 0.733917723952326, 0.8593156478277586),
              (0.2818813276944765, 0.5707599641163655, 0.7754914776368064),
              (0.20442906574394465, 0.29301038062283735, 0.35649365628604385)]

    # error plotting
    slopes_df["minus"] = slopes_df["slope"] - 1.96 * slopes_df["std_err"]
    slopes_df["plus"] = slopes_df["slope"] + 1.96 * slopes_df["std_err"]

    sns.set()
    # g = sns.relplot(x = "trial",y = "slope", col = "agent",kind = "line",hue = "rewsize",data = slopes_df, palette = colors)

    g = sns.FacetGrid(data = slopes_df, col="agent",hue = "rewsize", col_wrap=rollover,height=5,palette = colors)
    g = g.map(plt.plot,"trial","slope")
    g = (g.map(plt.fill_between,"trial", "minus", "plus",alpha = .3).add_legend().set_axis_labels("Trial", "Regression Coefficient between N0 and PRT"))

def mvt_plot(mvt_df,trial_range,agent_type):
    plt.figure()
    plt.title("Agent Marginal Value Theorem Estimation")
    colors = [(0.5430834294502115, 0.733917723952326, 0.8593156478277586),
              (0.2818813276944765, 0.5707599641163655, 0.7754914776368064),
              (0.20442906574394465, 0.29301038062283735, 0.35649365628604385)]

    trials_df = mvt_df[(np.isin(mvt_df["trial"], trial_range))].sort_values(["trial","timepoint"])
    x = range(len(trials_df))

    plt.plot(x,trials_df["instTrue"],label = "True Inst Rew")
    if agent_type == "MVT": # add option for RL here
        plt.plot(x,trials_df["instEst"],label = "Estimated Inst Rew")
    if agent_type == "Q":
        plt.plot(x,trials_df["v_patch"],label = "Estimated Patch Value")
    plt.plot(x,trials_df["avgEst"],label = "Estimated Avg Rew")

    rews = np.array(trials_df["rew"]) + .2 # adjust for ITI penalty
    rew_idx = np.where(rews > 0)[0]
    nonzero_rews = rews[rew_idx]
    color_list = []
    for i in range(len(nonzero_rews)):
        color_index = np.unique(nonzero_rews).tolist().index(nonzero_rews[i])
        color_list.append(colors[color_index])

    rew_idx = np.where(rews>0)[0]
    raster_loc = [-1.5 for i in range(len(rew_idx))]

    plt.scatter(rew_idx,raster_loc,marker = "|",s = 500,color = color_list,label = "Reward events")
    plt.legend()

    timeptArray = np.array(trials_df["timepoint"])
    # use plt.fill_between to signify different patches
    plt.fill_between(np.arange(len(trials_df)*4)/4,-1,4,where = np.repeat(np.roll(timeptArray > 0,-1),4),alpha = .2)
