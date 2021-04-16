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

TOTAL_SAMPLES = 1000000
HIDDEN_NETWORK_SIZE=16
SAMPLE_DIM = 16
BATCH_SIZE = 64
SCALE = 1

env = Game()

en_model = EN(env.n_actions, input_dim=HIDDEN_NETWORK_SIZE).to(device)
optimizer = optim.Adam(en_model.parameters(), lr=.01)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

agent = ActorLastLayer(env.n_actions, input_dim=HIDDEN_NETWORK_SIZE).to(device)
agent.load_weights(path="./models/history08/actor.pth")

def generate_samples():
    samples = torch.rand((TOTAL_SAMPLES, SAMPLE_DIM)).to(device)
    return samples * SCALE

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
valid_dataset = EmulatorNetworkDataset(samples[split1:])
test_dataset = EmulatorNetworkDataset(samples[split2:])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

train_losses = []
valid_losses = []
batch_number = []
def train():
    min_validate_loss = float("inf")
    for i, batch in enumerate(train_loader):
        target_actions = agent(batch).detach()
        actions = en_model(batch)
        squared_difference = torch.pow(target_actions - actions, 2)
        loss = torch.mean(torch.sum(squared_difference, dim=1))
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            validate_loss = validate()
            valid_losses.append(validate_loss.item())
            batch_number.append(i)
            print(f"VALIDATE_LOSS {i}: {validate_loss}")
            # scheduler.step()
            if validate_loss < min_validate_loss:
                torch.save(en_model.state_dict(), "en_model/en_model.pth")
            if validate_loss < min_validate_loss:
                min_validate_loss = validate_loss
    return min_validate_loss

def validate():
    with torch.no_grad():
        total_loss = 0
        for i, batch in enumerate(valid_loader):
            target_actions = agent(batch).detach()
            actions = en_model(batch)
            squared_difference = torch.pow(target_actions - actions, 2)
            loss = torch.mean(torch.sum(squared_difference, dim=1))
            total_loss += loss
        return total_loss / len(valid_loader)

def test():
    en_model.load_state_dict(torch.load("en_model/en_model.pth", map_location=torch.device("cpu")))
    with torch.no_grad():
        total_loss = 0
        for i, batch in enumerate(test_loader):
            target_actions = agent(batch).detach()
            actions = en_model(batch)
            squared_difference = torch.pow(target_actions - actions, 2)
            loss = torch.mean(torch.sum(squared_difference, dim=1))
            #if i % 1000 == 0:
            #    print(i, loss)
            total_loss += loss
        return total_loss / len(train_loader)

best_validation_loss = train()
test_loss = test()
print(f"BEST VALIDATION LOSS: {best_validation_loss}")
# en_model.load_state_dict(torch.load("models/en_models/en_damaged.pth", map_location=torch.device("cpu")))
print(f"TEST LOSS: {test_loss}")

fig, ax =plt.subplots(1,2)
ax1 = sns.lineplot(x=[i for i in range(len(train_losses))], y=train_losses, ax=ax[0])
ax1.set(xlabel="Batch #", ylabel="Loss", title="Training Set")
ax2 = sns.lineplot(x=batch_number, y=valid_losses, ax=ax[1])
ax2.set(xlabel="Batch #", ylabel="Loss", title="Validation Set")
plt.show()