import sys
sys.path.insert(1,'./tabQ')
sys.path.insert(1,'./deepQ')
from matplotlib import pyplot as plt
# custom imports
from TabQInterface import RLInterface
from deepQInterface import DeepRLInterface
from tabQAgents import Model1Agent,Model2Agent,Model3Agent,OmniscientAgent
from deepQAgents import DeepModel1Agent,DeepModel2Agent,DeepModel3Agent
from patchyEnvironment import PatchEnvironment
from performancePlots import mvt_plot,plot_prts,plot_rewrate,prt_bars,prt_hist,prt_plus_bars,mk_timecourse,plot_survival

# run RL interface
env = PatchEnvironment('probabilistic',nTimestates = 50,ITI_penalty = 2,timecost = .2)

agent_type = sys.argv[1]
nTrials = int(sys.argv[2])

# define agent w/ env parameters
nTimestates = env.nTimestates
rewsizes = env.rewsizes
nRewsizes = len(rewsizes)
maxRewsize = max(rewsizes)
if agent_type == 'Tab1':
    agent = Model1Agent(nRewsizes,"softmax",nTimestates,rewsizes,beta = 1.5)
elif agent_type == 'Tab2':
    agent = Model2Agent(nRewsizes,"softmax",nTimestates,rewsizes,beta = 1.5)
elif agent_type == 'Tab3':
    agent = Model3Agent(nRewsizes,"softmax",nTimestates,rewsizes,b = 1,a = 3,beta = 1.5)
elif agent_type == 'TabO':
    agent = OmniscientAgent(nRewsizes,"softmax",nTimestates,rewsizes,beta = 1.5)
elif agent_type == 'Deep1':
    agent = DeepModel1Agent(nTimestates,maxRewsize,"egreedy")
elif agent_type == 'Deep2':
    agent = DeepModel2Agent(nTimestates,maxRewsize,"softmax")
elif agent_type == 'Deep3':
    agent = DeepModel3Agent(maxRewsize,"egreedy",a = 1,b = 3)
else:
    raise ValueError("Please use Tab1, Tab2, Tab3, TabO, Deep1, Deep2, or Deep3 as agent arg")

if agent_type[:4] == 'Deep':
    rl = DeepRLInterface(agent,env)
else:
    rl = RLInterface(agent,env)

rl.run_trials(nTrials)

# --------- visualization --------- #
# rl.show_qtable()

vis_start = min(14500,nTrials-400)
filter_sd = 50

plot_prts(rl,filter_sd)

# assess agent efficiency
plot_rewrate(rl,filter_sd)

# mean PRT analysis
prt_bars(rl,filter_sd)
prt_hist(rl,vis_start)
prt_plus_bars(rl,vis_start)

# timecourse analysis
mk_timecourse(rl)
plot_survival(rl)

if agent_type[:4] == 'Deep':
    rl.plot_loss()
# plt.show()

if agent_type[:4] == 'Deep':
    mvt_plot(rl.mvt_df,range(vis_start-10,vis_start),"Q",deep = True)
else:
    mvt_plot(rl.mvt_df,range(vis_start-10,vis_start),"Q")

plt.show()
