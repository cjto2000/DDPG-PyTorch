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
if use_GPU:
    print("GPU power is available :)")

# gam environment
env = Game()

# decide hyperparameters
MAX_BUFF_SIZE = 10000
STATE_DIM = env.state_dim
N_ACTIONS = env.n_actions
LIMIT = env.limit
N_EPISODES = 10000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001
LOG_STEPS = 20
SAVE_STEPS = 100
THRESHOLD_STEPS = 1000

actor_model_path = "saved_models/actor.pth"
critic_model_path = "saved_models/critic.pth"

# initialize network models
actor_net = Actor(STATE_DIM, N_ACTIONS, LIMIT).cuda() if use_GPU else Actor(STATE_DIM, N_ACTIONS, LIMIT)
critic_net = Critic(STATE_DIM, N_ACTIONS).cuda() if use_GPU else Critic(STATE_DIM, N_ACTIONS)

target_actor_net = Actor(STATE_DIM, N_ACTIONS, LIMIT).cuda() if use_GPU else Actor(STATE_DIM, N_ACTIONS, LIMIT)
target_critic_net = Critic(STATE_DIM, N_ACTIONS).cuda() if use_GPU else Critic(STATE_DIM, N_ACTIONS)

if os.path.exists(actor_model_path):
    actor_net.load_state_dict(torch.load(actor_model_path))

if os.path.exists(critic_model_path):
    critic_net.load_state_dict(torch.load(critic_model_path))

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
    n_steps = 0
    while not is_done and n_steps < THRESHOLD_STEPS:
        S_var = Variable(torch.FloatTensor(S))
        A_pred = actor_net(S_var)
        A = A_pred.data.numpy() + torch.FloatTensor(noise)
        S_prime, R, is_done = env.take_action(A)
        # store transition in replay memory
        memory.add_to_memory((S, A, S_prime, R, is_done))
        # update the next state for next iteration
        S = S_prime
        R_total += R

        # Training on samples drawn from replay memory
        S_batch, A_batch, S_prime_batch, R_batch, is_done_batch = memory.sample(BATCH_SIZE)
        is_done_batch = is_done_batch.astype(int)

        # cast into variables
        S_batch = Variable(torch.FloatTensor(S_batch))
        A_batch = Variable(torch.FloatTensor(A_batch))
        S_prime_batch = Variable(torch.FloatTensor(S_prime_batch))
        R_batch = Variable(torch.FloatTensor(R_batch))
        is_done_batch = Variable(torch.FloatTensor(is_done_batch))

        if use_GPU:
            S_batch = S_batch.cuda()
            A_batch = A_batch.cuda()
            S_prime_batch = S_prime_batch.cuda()
            R_batch ().cuda()
            is_done_batch ().cuda()

        # update critic network -> Q(S, A)
        A_pred = target_actor_net(S_batch)
        target_y = R_batch + GAMMA * torch.mul(target_critic_net(S_batch, A_pred), 1 - is_done_batch)
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
        n_steps += 1

    return critic_loss, actor_loss, R_total

# history["rewards"] = []
history = {
    "rewards" : [],
    "critic_loss" : [],
    "actor_loss": [],
}

for i in range(N_EPISODES):
    noise = Orn_Uhlen(N_ACTIONS)
    l1, l2, R = train_one_episode(noise)
    # TODO: maintain running rewards and losses
    if i % LOG_STEPS == 0:
        history["rewards"].append(R)
        history["critic_loss"].append(l1)
        history["actor_loss"].append(l2)
        print("Episode %5d -- Rewards : %.5f -- Losses: %.5f(a)  %.5f(c)" %(i, R, l2, l1))
    if i % SAVE_STEPS == 0:
        torch.save(actor_net.state_dict(), actor_model_path)
        torch.save(critic_net.state_dict(), critic_model_path)

