import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import pickle
import os

from environment import Game
from utilities import soft_update
from model import Actor, Critic
from replay_mem import ReplayMemory
from utilities import soft_update, Orn_Uhlen

use_GPU = torch.cuda.is_available()

# gam environment
env = Game()

# decide hyperparameters
MAX_BUFF_SIZE = 100
STATE_DIM = env.state_dim
N_ACTIONS = env.n_actions
LIMIT = env.limit
N_EPISODES = 10000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.01

actor_model_path = "saved_models/actor.pth"
critic_model_path = "saved_models/critic.pth"

# initialize network models
actor_net = Actor(STATE_DIM, N_ACTIONS, LIMIT).cuda() if use_GPU else Actor(STATE_DIM, N_ACTIONS, LIMIT)
critic_net = Critic(STATE_DIM, N_ACTIONS).cuda() if use_GPU else Critic(STATE_DIM, N_ACTIONS)

target_actor_net = Actor(STATE_DIM, N_ACTIONS, LIMIT).cuda() if use_GPU else Actor(STATE_DIM, N_ACTIONS, LIMIT)
target_critic_net = Critic(STATE_DIM, N_ACTIONS).cuda() if use_GPU else Critic(STATE_DIM, N_ACTIONS)

if os.path.exists(actor_model_path):
    actor_net.load_state_dict(actor_model_path)

if os.path.exists(critic_model_path):
    critic_net.load_state_dict(critic_model_path)

target_actor_net.load_state_dict(actor_net.state_dict())
target_critic_net.load_state_dict(critic_net.state_dict())

# set optimizers
actor_optimizer = optim.Adam(actor_net.parameters())
critic_optimizer = optim.Adam(critic_net.parameters())

# Create replay memory
memory = ReplayMemory(MAX_BUFF_SIZE, STATE_DIM, N_ACTIONS)

def initialize_replay_mem():
    '''
    Initialize the replay memory.
    '''
    S = env.reset()
    # fill the buffer
    for _ in range(MAX_BUFF_SIZE):
        A = env.sample_action()
        S_prime, R, is_done = env.take_action(A)
        memory.add_to_memory((S, A, S_prime, R, is_done))
        if is_done:
            S = env.reset()
        else:
            S = S_prime

def train_one_episode(noise):
    S = env.reset()
    is_done = False
    R_total = 0
    while not is_done:
        S_var = Variable(torch.from_numpy(S).float())
        A_pred = actor_net(S_var)
        A = A_pred.data[0] + torch.from_numpy(noise).float()
        S_prime, R, is_done = env.take_action(A)
        # store transition in replay memory
        memory.add_to_memory((S, A, S_prime, R, is_done))
        # update the next state for next iteration
        S = S_prime
        R_total += R

        # Training on samples drawn from replay memory
        S_batch, A_batch, S_prime_batch, R_batch, is_done_batch = memory.sample(BATCH_SIZE)

        # cast into variables
        S_batch = torch.from_numpy(S_batch).float()
        A_batch = torch.from_numpy(A_batch).float()
        S_prime_batch = torch.from_numpy(S_prime_batch).float()
        R_batch = torch.from_numpy(R_batch).float()

        # update critic network -> Q(S, A)
        A_pred = target_actor_net(S_batch)
        target_y = R_batch + GAMMA * target_critic_net(S_batch, A_pred)
        y = critic_net(S_batch, A_batch)

        # prediction loss for critic
        critic_loss = torch.sum((target_y - y)**2)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # update actor network
        A_pred = actor_net(S_batch)
        actor_loss = critic_net(S_batch, A_pred)
        actor_loss = -1 * torch.sum(actor_loss)
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # soft_update target networks
        soft_update(critic_net.parameters(), target_critic_net.parameters(), TAU)
        soft_update(actor_net.parameters(), target_actor_net.parameters(), TAU)

    return R_total

# history["rewards"] = []
history = {
    "rewards" : []
}
for i in range(N_EPISODES):
    noise = Orn_Uhlen(N_ACTIONS)
    R = train_one_episode(noise)
    history["rewards"].append(R)
    if i % 100 == 0:
        print("Episode %5d -- Rewards : %5f" %(i, R))

