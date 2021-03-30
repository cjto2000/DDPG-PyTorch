import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CPN(nn.Module):
    def __init__ (self, state_dim, input_dim=400, output_dim=300):
        super(CPN, self).__init__()
        self.fc1 = nn.Linear(state_dim, input_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class ActorLastLayer(nn.Module):
    def __init__(self, action_dim, input_dim=300):
        super(ActorLastLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, action_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return x

    def load_weights(self, path):
        state_dict = torch.load(path)
        with torch.no_grad():
            self.fc1.weight.copy_(state_dict["fc3.weight"])
            self.fc1.bias.copy_(state_dict["fc3.bias"])

    def freeze_parameters(self):
        for param in self.fc1.parameters():
            param.requires_grad = False

class EN(nn.Module):
    def __init__(self, action_dim, input_dim=300):
        super(EN, self).__init__()
        self.fc1 = nn.Linear(input_dim, action_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return x

class Actor(nn.Module):
    # Actor provides the next action to take
    def __init__(self, state_dim, action_dim, limit, en=False):
        super(Actor, self).__init__()
        self.en = en
        self.limit = torch.FloatTensor(limit)

        self.fc1 = nn.Linear(state_dim, 400)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(400, 16)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(16, action_dim)
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        en_input = x
        x = F.tanh(self.fc3(x))
        if self.en:
            return x, en_input
        return x

class DamagedActor(nn.Module):
    def __init__(self, state_dim, action_dim, limit, hidden_dim=64, en=False):
        super(DamagedActor, self).__init__()
        self.en = en
        self.limit = torch.FloatTensor(limit)
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(hidden_dim, 16)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(16, action_dim)
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        en_input = x
        x = F.tanh(self.fc3(x))
        if self.en:
            return x, en_input
        return x
    
    def load_weights(self, path):
      state_dict = torch.load(path)
      state_dict['fc1.weight'] = state_dict['fc1.weight'][:self.hidden_dim, :]
      state_dict['fc1.bias'] = state_dict['fc1.bias'][:self.hidden_dim]
      state_dict['fc2.weight'] = state_dict['fc2.weight'][:, :self.hidden_dim]
      with torch.no_grad():
        self.fc1.weight.copy_(state_dict["fc1.weight"])
        self.fc1.bias.copy_(state_dict["fc1.bias"])
        self.fc2.weight.copy_(state_dict["fc2.weight"])
        self.fc2.bias.copy_(state_dict["fc2.bias"])
        self.fc3.weight.copy_(state_dict["fc3.weight"])
        self.fc3.bias.copy_(state_dict["fc3.bias"])

class ActorEN(nn.Module):
    # Actor provides the next action to take
    def __init__(self, state_dim, action_dim, limit):
        super(ActorEN, self).__init__()
        self.limit = torch.FloatTensor(limit)

        self.fc1 = nn.Linear(state_dim, 400)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(400, 300)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(300, action_dim)
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        en_input = x
        x = F.tanh(self.fc3(x))
        return x, en_input

class SuperLesionedActor(nn.Module):
    # Actor provides the next action to take
    def __init__(self, state_dim, action_dim, hidden_dim, limit):
        super(SuperLesionedActor, self).__init__()
        self.limit = torch.FloatTensor(limit)

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(hidden_dim, 16)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(16, action_dim)
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)

        self.cpn = CPN(state_dim, output_dim=16)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actor_input = x + self.cpn(state)
        x = F.tanh(self.fc3(actor_input))
        return x, actor_input

    def load_weights(self, path):
        state_dict = torch.load(path)
        with torch.no_grad():
            self.fc1.weight.copy_(state_dict["fc1.weight"])
            self.fc1.bias.copy_(state_dict["fc1.bias"])
            self.fc2.weight.copy_(state_dict["fc2.weight"])
            self.fc2.bias.copy_(state_dict["fc2.bias"])
            self.fc3.weight.copy_(state_dict["fc3.weight"])
            self.fc3.bias.copy_(state_dict["fc3.bias"])

    def load_en_weights(self, path):
        state_dict = torch.load(path)
        with torch.no_grad():
            self.fc3.weight.copy_(state_dict["fc1.weight"])
            self.fc3.bias.copy_(state_dict["fc1.bias"])

    def zero_weights(self, percent=.50):
        # zero first layer
        with torch.no_grad():
            shape = self.fc1.weight.shape
            print(shape[0], shape[1])
            num = int(shape[0] * percent)
            self.fc1.weight[0:num, :] = 0

    def freeze_parameters(self):
        #for param in self.fc1.parameters():
        #    param.requires_grad = False
        #for param in self.fc2.parameters():
        #    param.requires_grad = False
        for param in self.fc3.parameters():
            param.requires_grad = False

class LesionedActor(nn.Module):
    # Actor provides the next action to take
    def __init__(self, state_dim, action_dim, limit, en=False):
        super(LesionedActor, self).__init__()
        self.limit = torch.FloatTensor(limit)
        self.en = en

        self.fc1 = nn.Linear(state_dim, 400)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(400, 300)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(300, action_dim)
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)

        self.cpn = CPN(state_dim)

    def forward(self, x):
        cpn_input = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_cpn = F.relu(self.cpn(cpn_input))
        en_input = x + x_cpn
        x = F.tanh(self.fc3(x + x_cpn))
        if self.en:
            return x, en_input
        return x

    def load_weights(self, path="./good_models/actor.pth"):
        state_dict = torch.load(path)
        with torch.no_grad():
            self.fc1.weight.copy_(state_dict["fc1.weight"])
            self.fc1.bias.copy_(state_dict["fc1.bias"])
            self.fc2.weight.copy_(state_dict["fc2.weight"])
            self.fc2.bias.copy_(state_dict["fc2.bias"])
            self.fc3.weight.copy_(state_dict["fc3.weight"])
            self.fc3.bias.copy_(state_dict["fc3.bias"])

    def zero_weights(self, percent=.20):
        # zero first layer
        with torch.no_grad():
            shape = self.fc1.weight.shape
            print(shape[0], shape[1])
            num = int(shape[0] * percent)
            self.fc1.weight[0:num, :] = 0

    def freeze_parameters(self):
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = False
        for param in self.fc3.parameters():
            param.requires_grad = False


class Critic(nn.Module):
    # Critic estimates the state-value function Q(S, A)
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(400 + action_dim, 300)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(300, 1)
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)

    def forward(self, state, action):
        s = F.relu(self.fc1(state))
        x = torch.cat((s, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
