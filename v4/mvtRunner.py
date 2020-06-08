import sys
import numpy as np
from matplotlib import pyplot as plt
# custom imports
from mvtInterface import MVTInterface
from mvtAgents import MVT_agentAverager,MVT_agentDoubleDelta
from patchyEnvironment import PatchEnvironment
from performancePlots import mvt_plot

# run RL interface
env = PatchEnvironment('probabilistic',nTimestates = 50,ITI_penalty = 2,timecost = .2)

agent = MVT_agentAverager(env.ITI_penalty)

nTrials = int(sys.argv[1])

mvt = MVTInterface(agent,env)

mvt.run_trials(nTrials)

# --------- visualization --------- #
vis_start = 400
filter_sd = 50

mvt.plot_prts(filter_sd)

# assess agent efficiency
mvt.plot_rewrate(filter_sd)

# mean PRT analysis
mvt.prt_bars(filter_sd)
mvt.prt_hist(vis_start)
mvt.prt_plus_bars(vis_start)

# timecourse analysis
mvt.mk_timecourse()
mvt.plot_survival()

plt.figure()
plt.plot(agent.avg_list,label = "Environmental Reward Rate Estimate")
plt.legend()
plt.title('Environmental reward rate estimation over time')

mvt_plot(mvt.mvt_df,range(400,410),"MVT")

plt.show()
