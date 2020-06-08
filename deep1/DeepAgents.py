import numpy as np
from numpy import random as rnd
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

# just a feed forward neural network to estimate Q(s,a) values
class DQN(nn.Module):
    def __init__(self, envstate_dim, action_dim):
        super(DQN, self).__init__()
        self.input_dim = envstate_dim
        self.output_dim = action_dim

        self.ff = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 124),
            nn.ReLU(),
            nn.Linear(124, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim),
#             nn.Softmax(dim=0)
        )

    def forward(self, state):
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

# a class for agents that use feedforward neural networks to calculate Q(s,a)
class DeepAgent():
    def __init__(self, board_size):
        self.policy_net = DQN(board_size,board_size) # network used to calculate policy
        self.target_net = DQN(board_size,board_size) # network used to calculate target
        self.target_net.eval() # throw that baby in eval mode because we don't care about its gradients
        self.target_update = 50 #
        self.replay_buffer = Buffer() # replay buffer implemented as a list
        self.eps_start = 0.1 # exploration rate
        self.eps_end = 0.95
        self.eps_decay = 300
        self.epsilon = self.eps_start
        self.gamma = 0.99 # discount

        self.optimizer = torch.optim.SGD(self.policy_net.parameters(),lr=0.01, momentum=0.9)
#         self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())
#         self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.huber_loss = F.smooth_l1_loss

    def select_action(self,state):
        state = torch.FloatTensor(state).float()
        if rnd.rand() < self.epsilon: # greedy action
            with torch.no_grad():
                qvals = self.policy_net.forward(state) # forward run through the policy network
                action = np.argmax(qvals.detach().numpy()) # need to detach from auto_grad before sending to numpy
        else:
            action = random.choice(list(range(board_size)))
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
