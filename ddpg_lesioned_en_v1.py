import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os
import numpy as np

from constants import *
from model import SuperLesionedActor, Critic, EN

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {device}")


class Orn_Uhlen:
    def __init__(self, n_actions, mu=0, theta=0.15, sigma=0.2):
        self.n_actions = n_actions
        self.X = np.ones(n_actions) * mu
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

    def reset(self):
        self.X = np.ones(self.n_actions) * self.mu

    def sample(self):
        dX = self.theta * (self.mu - self.X)
        dX += self.sigma * np.random.randn(self.n_actions)
        self.X += dX
        return self.X


class DDPG:
    def __init__(self, env, memory):
        self.env = env

        n_inp = env.state_dim
        n_out = env.n_actions
        a_limit = env.limit

        self.en_net = EN(n_out).to(device)
        self.en_net.load_state_dict(torch.load("damaged_models/en.pth"))
        self.en_net.freeze_parameters()

        self.actor_net = SuperLesionedActor(n_inp, n_out, a_limit, en=True).to(device)
        self.actor_net.freeze_parameters()
        self.actor_net.load_weights()
        self.critic_net = Critic(n_inp, n_out).to(device)
        self.critic_net.load_state_dict(torch.load("damaged_models/critic.pth"))

        self.target_actor_net = SuperLesionedActor(n_inp, n_out, a_limit).to(device)
        self.target_critic_net = Critic(n_inp, n_out).to(device)

        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=A_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=C_LEARNING_RATE)

        self.memory = memory
        self.noise = Orn_Uhlen(n_out)

    def train_one_episode(self, batch_size=32):
        S = self.env.reset()
        is_done = False
        R_total = 0
        n_steps = 0
        while not is_done and n_steps < THRESHOLD_STEPS:
            self.env.render()
            S_var = Variable(torch.FloatTensor(S)).unsqueeze(0).to(device)
            A_pred, en_input = self.actor_net(S_var)
            A_pred = A_pred.detach()
            noise = self.noise.sample()
            A = (A_pred.data.cpu().numpy() + noise)[0][:]

            S_prime, R, is_done = self.env.take_action(A)
            # store transition in replay memory
            self.memory.add_to_memory((S, A, S_prime, R, is_done))
            # update the next state for next iteration
            S = S_prime
            R_total += R

            # Training on samples drawn from replay memory
            S_batch, A_batch, S_prime_batch, R_batch, is_done_batch = self.memory.sample(batch_size)

            # cast into variables
            S_batch = Variable(torch.FloatTensor(S_batch)).to(device)

            # find actor loss
            A_actor, en_input = self.actor_net(S_batch)
            A_en = self.en_net(en_input)
            actor_loss = -1 * torch.mean(self.critic_net(S_batch, A_en))

            # update actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            n_steps += 1

        self.noise.reset()
        return actor_loss, R_total, n_steps