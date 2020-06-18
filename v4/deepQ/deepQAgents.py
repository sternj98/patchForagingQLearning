import numpy as np
from numpy import random as rnd
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

STAY = 0
LEAVE = 1

OFF = 0
ON = 1

# just a feed forward neural network to estimate Q(s,a) values
# can probably be pretty simple
class DQN(nn.Module):
    def __init__(self, envstate_dim, action_dim):
        super(DQN, self).__init__()
        self.input_dim = envstate_dim
        self.output_dim = action_dim

        self.ff = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32), # single hidden layer can maybe improve approx
            nn.Tanh(),
            nn.Linear(32, self.output_dim),
        )

    def forward(self, state):
        state = torch.FloatTensor(state).float()
        # print(state)
        qvals = self.ff(state)
        return qvals

# replay buffers implemented as lists
class Buffer():
    def __init__(self):
        self.buffer = []

    def size(self):
        return len(self.buffer)

    def push(self,state,action,new_state,reward):
        experience = (state,action,new_state,reward)
        self.buffer.append(experience)

    def sample(self,batch_size):
        batchSample = random.sample(self.buffer,batch_size)
        # now need to put everyone in the correct columns
        state_batch = []
        action_batch = []
        new_state_batch = []
        reward_batch = []

        # prepare the batch sample for training
        for experience in batchSample:
            state,action,new_state,reward = experience
            state_batch.append(state)
            action_batch.append(action)
            new_state_batch.append(new_state)
            reward_batch.append(reward)
        return (state_batch, action_batch, reward_batch, new_state_batch)

class DeepQAgent():
    """
        General class for RL agents that use FF nns to est. Q(s,a)
        the envstate: [patchOn,patchOff,rewsize/maxrewsize,intValue/maxIntValue]
        # add softmax
    """
    def __init__(self,decision_type):
        envstate_dim = 4
        action_dim = 2
        self.policy_net = DQN(envstate_dim, action_dim) # network used to calculate policy
        self.target_net = DQN(envstate_dim, action_dim) # network used to calculate target
        self.target_net.eval() # throw that baby in eval mode because we don't care about its gradients
        self.target_update = 50**2 # this works really poorly
        self.replay_buffer = Buffer() # replay buffer implemented as a list
        self.decision_type = decision_type
        # dynamic params for egreedy
        self.eps_start = 0.1 # exploration rate
        self.eps_end = 0.95
        self.eps_decay = 300
        self.epsilon = self.eps_start
        # dynamic params for softmax
        self.beta0 = .25
        self.beta_final = 1.0
        self.beta_decay = 700
        self.beta = self.beta0

        self.gamma = 0.7 # discount

        # self.optimizer = torch.optim.SGD(self.policy_net.parameters(),lr=0.01, momentum=0.9)
        # self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(),lr = 0.0005)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),lr = 0.00005)
        self.huber_loss = F.smooth_l1_loss # nn.L1Loss(reduction = "mean")

    def select_action(self,state):
        # state = torch.FloatTensor(state).float()
        if self.decision_type == "egreedy":
            if rnd.rand() < self.epsilon: # greedy action
                with torch.no_grad():
                    qvals = self.policy_net.forward(state) # forward run through the policy network
                    action = np.argmax(qvals.detach().numpy()) # need to detach from auto_grad before sending to numpy
            else:
                action = random.choice([STAY,LEAVE])
        elif self.decision_type == "softmax":
            Q_stay,Q_leave = self.policy_net.forward(state).detach().numpy()
            p_stay = (1 + np.exp(-self.beta * (Q_stay - Q_leave))) ** (-1)
            return STAY if rnd.rand() < p_stay else LEAVE
        return action

    def update(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return
        batch = self.replay_buffer.sample(batch_size)

        self.optimizer.zero_grad() # zero_grad before computing loss

        loss = self.compute_loss(batch)
        loss.backward() # perform back propagation

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return loss

    def compute_loss(self, batch):
        states, actions, rewards, next_states = batch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        curr_Q = self.policy_net.forward(states).gather(1,actions.unsqueeze(1)) # calculate the current Q(s,a) estimates
        next_Q = self.target_net.forward(next_states) # calculate Q'(s,a) (EV)
        max_next_Q = torch.max(next_Q,1)[0] # equivalent of taking a greedy action
        expected_Q = rewards + self.gamma * max_next_Q # Calculate total Q(s,a)

#         print(curr_Q.size())
#         print(expected_Q.size())
        loss = self.huber_loss(curr_Q, expected_Q.unsqueeze(1))
        return loss

# normalize before or after??
class DeepModel1Agent(DeepQAgent):
    """
        rew integration is a function of time
    """
    def __init__(self,nTimestates,maxRewsize,decision_type):
        super().__init__(decision_type)
        self.model = "Model1"
        self.nTimestates = nTimestates
        self.maxRewsize = maxRewsize

    def integrate(self,env_state):
        if env_state["patch"] == ON:
            return [1, 0, env_state["rewsize"],env_state["t"] / self.nTimestates] # normalize
        else:
            return [0,1,0,0]

class DeepModel2Agent(DeepQAgent):
    """
        rew integration is a function of time since previous reward, reward size
    """
    def __init__(self,nTimestates,maxRewsize,decision_type):
        super().__init__(decision_type)
        self.model = "Model2"
        self.nTimestates = nTimestates
        self.maxRewsize = maxRewsize

    def integrate(self,env_state):
        if env_state["patch"] == ON:
            time_since = list(reversed(env_state["rews"][:(env_state["t"]+1)])).index(env_state["rewsize"])
            return [1, 0,env_state["rewsize"]/self.maxRewsize, time_since / self.nTimestates] #
        else:
            return [0,1,0,0]

class DeepModel3Agent(DeepQAgent):
    """
        rew integration is a function of time since previous reward, reward size
    """
    def __init__(self,maxRewsize,decision_type,a = 3,b = 1):
        super().__init__(decision_type)
        self.model = "Model3"
        self.intNorm = 10 * (a - b) # will almost never get 10 rewards
        self.maxRewsize = maxRewsize
        self.a = a
        self.b = b

    def integrate(self,env_state):
        if env_state["patch"] == ON:
            t = env_state["t"]
            nRews = sum(env_state["rews"][:(t+1)])/env_state["rewsize"]
            rew_int = self.a * nRews - self.b * t
            return [1,0,env_state["rewsize"]/self.maxRewsize, rew_int / self.intNorm]
        else:
            return [0,1,0,0]

# here add model 4, 5, and Jan's inference model
