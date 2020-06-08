from numpy import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import scipy.io as sio

from QLearner import RLInterface
from patchForagingAgents import Model1Agent,Model2Agent,Model3Agent
from patchyEnvironment import PatchEnvironment,generate_pdfs

session = 'data/ql80_20200317.mat'

data = sio.loadmat(session)
prt_n0_rews = data['prt_n0_rewbarcodes']

prts = prt_n0_rews[:,0]
n0 = prt_n0_rews[:,1]
pdfs = generate_pdfs(np.unique(n0))
rews = prt_n0_rews[:,2:]
nTimestates = rews.shape[1]

# now need to go through, find -1s, and replace with rewards
for trial in range(rews.shape[0]):
    curr_rews = rews[trial,:]
    curr_rewsize = rews[trial,0]
    curr_n0 = n0[trial]
    prt_fl = int(np.floor(prts[trial]))
    curr_rewlocs = np.where(rnd.random(nTimestates) - pdfs[curr_n0][:nTimestates] < 0)[0]
    new_curr_rewlocs = curr_rewlocs[np.where(curr_rewlocs > prt_fl)[0]] # only change after leaving
    curr_rews[new_curr_rewlocs] = curr_rewsize
    curr_rews[np.where(curr_rews < 0 )[0]] = 0 # change the -1s
    rews[trial,:] = curr_rews

# now generate session with list comprehension
session = [{"rews":rews[trial,:] , "n0" : n0[trial]} for trial in range(len(n0))]

env = PatchEnvironment('probabilistic',nTimestates = 50,ITI_penalty = 2,timecost = .2)
agent_type = sys.argv[1]
if agent_type == 'Model1':
    agent = Model1Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,beta = 1.5)
if agent_type == 'Model2':
    agent = Model2Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,beta = 1.5)
if agent_type == 'Model3': # model 3 biases towards longer prts... why?
    agent = Model3Agent(len(env.rewsizes),"softmax",env.nTimestates,env.rewsizes,b = 1,a = 3,beta = 1.5)

rl = RLInterface(agent,env)

rl.run_trials(5000) # overtrain the agent
rl.prt_plus_bars(10)
plt.show() # check behavior

# run the behavior
rl.run_trials(500,probe_specs = session)

for rewsize in rl.prts_plus.keys():
    means = []
    for n0 in rl.prts_plus[rewsize].keys():
        means.append(np.mean(rl.prts_plus[rewsize][n0]))
    print("%i uL mean prts:\n"%rewsize , means)

errors = np.array(rl.prts_list) - np.array(prts)
print(errors)
jit_qprts = rl.prts_list + rnd.random(len(prts))*.2
d = {'trials': list(range(len(rl.prts_list))),
     'jit_qprts': jit_qprts, 'true_prts': prts,'errors': errors}
df = pd.DataFrame(data=d)

plt.figure()
plt.subplot(1,3,1)
plt.title('Jittered QLearner PRTs Colored by Error Magnitude')
sns.scatterplot(x = 'trials',y = 'jit_qprts',hue = 'errors',data = df)
plt.subplot(1,3,2)
plt.title('True PRTs Colored by Error Magnitude')
sns.scatterplot(x = 'trials',y = 'true_prts',hue = 'errors',data = df)
plt.subplot(1,3,3)
plt.title('QLearner PRT - True PRT')
sns.scatterplot(x = 'trials',y = 'errors',data = df)
plt.show()

print("Mean squared error:" , np.mean(np.array((prts - rl.prts_list))**2))
