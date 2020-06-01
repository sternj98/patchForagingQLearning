import random
from numpy import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress as linregress

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

def plot_prt_slopes(prt_df):
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

    g = sns.FacetGrid(data = slopes_df, col="agent",hue = "rewsize", col_wrap=2,height=5,palette = colors)
    g = g.map(plt.plot,"trial","slope")
    g = (g.map(plt.fill_between,"trial", "minus", "plus",alpha = .3).add_legend().set_axis_labels("Trial", "Regression Coefficient between N0 and PRT"))
