import numpy as np
from scipy.stats import gamma
from matplotlib import pyplot as plt
import seaborn as sns

def m1(env_state):
    return env_state["t"]

def m2(env_state):
    time_since = list(reversed(env_state["rews"][:(env_state["t"]+1)])).index(env_state["rewsize"])
    return time_since

def m3(env_state):
    a = 3
    b = 1
    t = env_state["t"]
    rew_int = a * sum(env_state["rews"][:(t+1)])/env_state["rewsize"] - b * t
    return int(rew_int)

def m4(env_state):
    # mean n0 is .25, mean of gamma is a0 / b0 (n/lambda)
    a0 = 1
    b0 = 4
    tau = 8
    t = env_state["t"]
    z = np.sum(env_state["rews"][:(t+1)])/env_state["rewsize"]
    return (a0 + z) / (b0 + tau * (1 - np.exp(-t/tau)))

def m5(env_state):
    x = 3
    a = 2
    b = 1
    t = env_state["t"]
    return a * np.sum(env_state["rews"][:(t)])/env_state["rewsize"] + env_state["rews"][t] * x - b * t

state = {"rews" : [1,0,0,0,1,0,0,1,0,1,0,1,0] , "t" : 0,"rewsize":1}

m1list = []
m2list = []
m3list = []
m4list = []
m5list = []

for i in range(len(state["rews"])):
    m1list.append(m1(state))
    m2list.append(m2(state))
    m3list.append(m3(state))
    m4list.append(m4(state))
    m5list.append(m5(state))

    state["t"] += 1


x = np.linspace(0,1,100)
dist = gamma.pdf(1,4,x)
plt.plot(x,dist)
plt.title("Gamma prior")
plt.xlabel('N0')
plt.ylabel('Probability Mass')
plt.show()

plt.figure()
# plt.plot(np.array(m1list) / sum(m1list))
plt.title('Standardized Reward Integration for Sequence: [1,0,0,0,1,0,0,1,0,1,0,1,0]')
plt.plot(-np.array(m2list) / np.std(m2list),label = "Memoryless Integrator")
plt.plot(np.array(m3list) / np.std(m3list),label = "Basic Integrator (a=3, b=1)")
plt.plot(np.array(m4list) / np.std(m4list),label = "Bayesian Estimation of N0 (a0=1, b0=4)")
plt.plot(np.array(m5list) / np.std(m5list),label = "Recency-Biased Integrator (a=1, b=2, x=3)")
plt.legend()
plt.show()
