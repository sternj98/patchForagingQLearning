import random
from numpy import random as rnd
import matplotlib.pyplot as plt
import numpy as np

def BIC(ll):
    """
        Calculate the Bayesian Information Criterion from log-likelihood
    """
    return np.log(n) * k - 2 * np.log(ll)
    # n is the sample size
    # k is the number of parameters



# this is actually going to be somewhat involved...
# need to have a new RL mode:
# - learning is (off?)
# - if env["t"] == true_prt[trial]: add the softmax value of leave to the ll
# - then just run normally
# - we might need to force the QLearner to take some ghost steps... actions it wouldn't have taken
#   if the mouse had not have been on the patch at that timestep
