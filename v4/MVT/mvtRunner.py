import sys
import numpy as np
from matplotlib import pyplot as plt
# custom imports
from mvtInterface import MVTInterface
from mvtAgents import MVT_agentAverager,MVT_agentDoubleDelta
from patchyEnvironment import PatchEnvironment
from performancePlots import mvt_plot,plot_prts,plot_rewrate,prt_bars,prt_hist,prt_plus_bars,mk_timecourse,plot_survival

# run RL interface
env = PatchEnvironment('probabilistic',nTimestates = 50,ITI_penalty = 2,timecost = .2)

# agent = MVT_agentDoubleDelta(env.ITI_penalty)
agent = MVT_agentDoubleDelta(env.ITI_penalty)

nTrials = int(sys.argv[1])

mvt = MVTInterface(agent,env)

mvt.run_trials(nTrials)

# --------- visualization --------- #
vis_start = 400
filter_sd = 50

plot_prts(mvt,filter_sd)

# assess agent efficiency
plot_rewrate(mvt,filter_sd)

# mean PRT analysis
prt_bars(mvt,filter_sd)
prt_hist(mvt,vis_start)
prt_plus_bars(mvt,vis_start)

# timecourse analysis
mk_timecourse(mvt)
plot_survival(mvt)

plt.figure()
plt.plot(agent.avg_list,label = "Environmental Reward Rate Estimate")
plt.legend()
plt.title('Environmental reward rate estimation over time')

mvt_plot(mvt.mvt_df,range(400,410),"MVT")

plt.show()
