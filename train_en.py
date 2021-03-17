import torch
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import os
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
from environment import Game
from collections import OrderedDict
from replay_mem import ReplayMemoryEN
from ddpg import *
from constants import *
from model import *
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler

TOTAL_SAMPLES = 5000000
HIDDEN_NETWORK_SIZE=300
SAMPLE_DIM = 24
BATCH_SIZE = 64

env = Game()

en_model = EN(env.n_actions, input_dim=HIDDEN_NETWORK_SIZE).to(device)
optimizer = optim.Adam(en_model.parameters(), lr=.01)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
file = "dim.json"

agent = Actor(env.state_dim, env.n_actions, env.limit, en=True).to(device)
agent.load_state_dict(torch.load("./models/good_models/actor.pth"))

def generate_samples():
    samples = torch.zeros((TOTAL_SAMPLES, SAMPLE_DIM)).to(device)
    with open(file) as f:
        dimensions = json.load(f)
    for i in range(SAMPLE_DIM):
        maxi = dimensions[str(i)]["max"]
        mini = dimensions[str(i)]["min"]
        dim_samples = (maxi - mini) * torch.rand((TOTAL_SAMPLES, 1)) + mini
        samples[:,i:i + 1] = dim_samples.to(device)
    return samples


samples = generate_samples()


class EmulatorNetworkDataset(Dataset):
    """Samples for EN training"""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


split1 = int(TOTAL_SAMPLES * .7)
split2 = int(TOTAL_SAMPLES * .8)
train_dataset = EmulatorNetworkDataset(samples[:split1])
valid_dataset = EmulatorNetworkDataset(samples[split1:split2])
test_dataset = EmulatorNetworkDataset(samples[split2:])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

train_losses = []
valid_losses = []
def train():
    min_validate_loss = 100
    for i, batch in enumerate(train_loader):
        target_actions, en_input = agent(batch)
        target_actions = target_actions.detach()
        en_input = en_input.detach()
        actions = en_model(en_input)
        squared_difference = torch.pow(target_actions - actions, 2)
        loss = torch.mean(torch.sum(squared_difference, dim=1))
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            validate_loss = validate()
            valid_losses.append(validate_loss.item())
            if validate_loss < min_validate_loss:
              min_validate_loss = validate_loss
              print(f"MIN VALIDATE LOSS IS {min_validate_loss}")
              if min_validate_loss < .2:
                torch.save(en_model.state_dict(), "en_model/en.pth") 
            #if validate_loss < 1.5:
            #  scheduler.step() 
        if i % 1000 == 0:
            print(i, loss)

def validate():
    print("validating")
    with torch.no_grad():
        total_loss = 0
        for i, batch in enumerate(valid_loader):
            target_actions, en_input = agent(batch)
            target_actions = target_actions.detach()
            en_input = en_input.detach()
            actions = en_model(en_input)
            squared_difference = torch.pow(target_actions - actions, 2)
            loss = torch.mean(torch.sum(squared_difference, dim=1))
            total_loss += loss
        return total_loss / len(valid_loader)

def test():
    with torch.no_grad():
        total_loss = 0
        for i, batch in enumerate(test_loader):
            target_actions, en_input = agent(batch)
            target_actions = target_actions.detach()
            en_input = en_input.detach()
            actions = en_model(en_input)
            squared_difference = torch.pow(target_actions - actions, 2)
            loss = torch.mean(torch.sum(squared_difference, dim=1))
            if i % 1000 == 0:
                print(i, loss)
            total_loss += loss
        return total_loss / len(train_loader)

train()
print(test())

fig, ax =plt.subplots(1,2)
sns.lineplot(x=[i for i in range(len(train_losses))], y=train_losses, ax=ax[0])
sns.lineplot(x=[i for i in range(len(valid_losses))], y=valid_losses, ax=ax[1])
plt.show()

