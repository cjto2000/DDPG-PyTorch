import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# xavier init from https://github.com/transedward/pytorch-ddpg/blob/master/agents/ddpg_low_dim.py
def xavier_init(tensor):
    fanin = tensor.size(1)
    v = 1.0 / np.sqrt(fanin)
    nn.init.uniform_(tensor, -v, v)

class Actor(nn.Module):
    # Actor provides the next action to take
    def __init__(self, state_dim, action_dim, limit):
        super(Actor, self).__init__()
        self.limit = torch.FloatTensor(limit)

        self.fc1 = nn.Linear(state_dim, 256)
        xavier_init(self.fc1.weight)

        self.fc2 = nn.Linear(256, 128)
        xavier_init(self.fc2.weight)

        self.fc3 = nn.Linear(128, 64)
        xavier_init(self.fc3.weight)

        self.fc4 = nn.Linear(64, action_dim)
        xavier_init(self.fc4.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.tanh(self.fc4(x))
        return x * self.limit


class Critic(nn.Module):
    # Critic estimates the state-value function Q(S, A)
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1_s = nn.Linear(state_dim, 256)
        xavier_init(self.fc1_s.weight)

        self.fc2_s = nn.Linear(256, 128)
        xavier_init(self.fc2_s.weight)

        self.fc1_a = nn.Linear(action_dim, 128)
        xavier_init(self.fc1_a.weight)

        self.fc3 = nn.Linear(256, 128)
        xavier_init(self.fc3.weight)

        self.fc4 = nn.Linear(128, 1)
        xavier_init(self.fc4.weight)

    def forward(self, state, action):
        s = F.relu(self.fc1_s(state))
        s = F.relu(self.fc2_s(s))

        a = F.relu(self.fc1_a(action))

        x = torch.cat((s, a), dim=1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
