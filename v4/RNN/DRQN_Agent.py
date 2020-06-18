import numpy as np
from numpy import random as rnd
import random
from matplotlib import pyplot as plt
import torch
from DRQN_utils import GruNet,SeqBuffer
import torch.nn as nn

# a class for agents that use feedforward neural networks to calculate Q(s,a)
class DeepRQAgent():
    def __init__(self,state_dim,action_dim,episode_len):
        self.policy_net = GruNet(state_dim,action_dim) # network used to calculate policy
        self.target_net = GruNet(state_dim,action_dim) # network used to calculate target
        self.target_net.eval() # throw that baby in eval mode because we don't care about its gradients
        self.target_update = 100 # update our target network every 50 timesteps
        self.replay_buffer = SeqBuffer(1000,episode_len) # sequential replay buffer implemented as list of lists
        self.action_dim = action_dim

        self.eps_start = 0.1 # initial exploration rate
        self.eps_end = 0.95 # ultimate exploration value
        self.eps_decay = 200 # decay parameter for exploration rate
        self.epsilon = self.eps_start # initialize epsilon

        self.gamma = 0.99 # discount

        # update these values with TD learning
        self.value_leave = 0
        self.Q_ITI = [0,0]

#         self.optimizer = torch.optim.SGD(self.policy_net.parameters(),lr=0.001, momentum=0.9)
#         self.optimizer = torch.optim.RMSprop(self.policy_net.parameters()) # experiment w/ different optimizers
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),lr = .001)
        self.huber_loss = F.smooth_l1_loss

    def select_action(self,states):
        state = torch.FloatTensor(states)
        if rnd.rand() < self.epsilon: # greedy action
            with torch.no_grad():
                value_stay = self.policy_net.forward(state.unsqueeze(0).unsqueeze(2)) # forward run through the policy network
                if value_stay > self.value_leave:
                    return STAY
                else:
                    return LEAVE
                # action = np.argmax(qvals.detach().numpy()) # need to detach from auto_grad before sending to numpy
        else:
            action = random.choice([STAY,LEAVE])
        return action


    def update(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return
        batch = self.replay_buffer.sample(batch_size)

        self.optimizer.zero_grad() # zero_grad before computing loss

        loss = self.compute_loss(batch)

        loss.backward() # get the gradients

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1) # gradient clipping

        self.optimizer.step() # backpropagate

        return loss

    def compute_loss(self, batch):
        states, actions, rewards, next_states = batch
        states = states
        actions = actions
        rewards = rewards
        next_states = next_states

        curr_Q = self.policy_net.forward(states).gather(1,actions.unsqueeze(1)) # calculate the current Q(s,a) estimates
        next_Q = self.target_net.forward(next_states.unsqueeze(1).unsqueeze(2)) # calculate Q'(s,a) (EV)
        max_next_Q = torch.max(next_Q,1)[0] # equivalent of taking a greedy action
        expected_Q = rewards + self.gamma * max_next_Q # Calculate total Q(s,a)

        loss = self.huber_loss(curr_Q, expected_Q.unsqueeze(1)) # unsqueeze is really important here to match dims!
        return loss
