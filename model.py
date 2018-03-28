import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, limit):
        super(Actor, self).__init__()
        self.limit = limit

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x * self.limit


class Critic(nn.Module):
    # Critic estimates the state-value function Q(S, A)
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(action_dim, 128)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, state, action):
        s = F.relu(self.fc1(state))
        s = F.relu(self.fc2(s))

        a = F.relu(self.fc3(action))

        x = torch.cat((s, a), dim=1)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        return x
