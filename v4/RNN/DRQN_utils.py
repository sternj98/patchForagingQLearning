class GruNet(nn.Module):
    def __init__(self, env_dim, nActions,hidden_dim = 124):
        super(GruNet, self).__init__()
        self.hidden_dim = hidden_dim

        # The GRU takes states as inputs, and outputs hidden states with dimensionality hidden_dim
        self.gru = nn.GRU(env_dim, hidden_dim,batch_first = True)

        # The linear layer that maps from hidden state space to action space
        self.hidden2action = nn.Linear(hidden_dim, nActions)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        gru_out, hidden = self.gru(x,hidden)
        gru_out = gru_out[:,-1,:] # take the last output
        action_space = self.hidden2action(gru_out)
        return action_space

    def init_hidden(self, batch_size):
        return torch.zeros(1,batch_size, self.hidden_dim, dtype=torch.float)

# sequential replay buffer
class SeqBuffer():
    def __init__(self,capacity,episode_len,seq_length = 10):
        self.memory = []
        self.capacity = capacity
        self.episode_len = episode_len # currently working with fixed length episode... dont even need this!
        self.seq_length = seq_length

    def size(self):
        return len(self.memory)

    # add a memory *** of an episode ***
    def push(self,states,actions,new_states,rewards):
        transition = [states,actions,new_states,rewards]
        self.memory.append(transition)
        if self.size() > self.capacity:
            del self.memory[0]

    # take a random sample of batch_size sequences to perform learning on decorrelated transition sequences
    def sample(self,batch_size):
        eps = random.sample(range(self.size()),batch_size)
        # random.choices to allow for replacement
        ends = random.choices(range(self.episode_len), k=batch_size) # can change this to allow for variable len episodes
        begins = [max(x-self.seq_length,0) for x in ends]
        samp = []

        state_batch = []
        action_batch = []
        new_state_batch = []
        reward_batch = []

        for ep, begin, end in zip(eps, begins, ends):
            states,actions,new_states,rewards = [x[begin:end+1] for x in self.memory[ep]]

            state_batch.append(states) # keep as list of lists now, pad + pack at end when we know lengths
            action_batch.append(actions[-1])
            new_state_batch.append(new_states[-1])
            reward_batch.append(rewards[-1])

        seq_lengths = torch.LongTensor(list(map(len, state_batch)))

        rev_state_batch = [torch.FloatTensor(x[::-1]) for x in state_batch]
        padded_states = pad_sequence(rev_state_batch, batch_first=True, padding_value=-1)
        # flip to get 0s in beginning of sequence
        padded_states = torch.from_numpy(np.flip(padded_states.numpy(),axis = 1).copy())
#         packed_states = pack_padded_sequence(padded_states.unsqueeze(2),seq_lengths.cpu().numpy(),batch_first=True,enforce_sorted = False)
        padded_states = padded_states.unsqueeze(2)
        return (padded_states, torch.LongTensor(action_batch), torch.FloatTensor(reward_batch), torch.FloatTensor(new_state_batch))
