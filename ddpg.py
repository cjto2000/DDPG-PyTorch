import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import os

from constants import *
from model import Actor, Critic

class DDPG:
    def __init__(self, env, memory):
        self.env = env

        n_inp = env.state_dim
        n_out = env.n_actions
        a_limit = env.limit

        self.actor_net = Actor(n_inp, n_out, a_limit)
        self.critic_net = Critic(n_inp, n_out)

        self.target_actor_net = Actor(n_inp, n_out, a_limit)
        self.target_critic_net = Critic(n_inp, n_out)

        if os.path.exists(actor_model_path):
            self.actor_net.load_state_dict(torch.load(actor_model_path))

        if os.path.exists(critic_model_path):
            self.critic_net.load_state_dict(torch.load(critic_model_path))

        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=A_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=C_LEARNING_RATE)

        self.memory = memory

    def train_one_episode(self, batch_size=64, noise=None):
        S = self.env.reset()
        is_done = False
        R_total = 0
        n_steps = 0
        while not is_done and n_steps < THRESHOLD_STEPS:
            S_var = Variable(torch.FloatTensor(S))
            A_pred = self.actor_net(S_var).detach()
            A = A_pred.data.numpy() + torch.FloatTensor(noise)

            S_prime, R, is_done = self.env.take_action(A)
            # store transition in replay memory
            self.memory.add_to_memory((S, A, S_prime, R, is_done))
            # update the next state for next iteration
            S = S_prime
            R_total += R

            # Training on samples drawn from replay memory
            S_batch, A_batch, S_prime_batch, R_batch, is_done_batch = self.memory.sample(batch_size)
            is_done_batch = is_done_batch.astype(int)

            # cast into variables
            S_batch = Variable(torch.FloatTensor(S_batch))
            A_batch = Variable(torch.FloatTensor(A_batch))
            S_prime_batch = Variable(torch.FloatTensor(S_prime_batch))
            R_batch = Variable(torch.FloatTensor(R_batch))
            is_done_batch = Variable(torch.FloatTensor(is_done_batch))

            A_critic = self.target_actor_net(S_prime_batch)
            Q_Spr_A = self.target_critic_net(S_prime_batch, A_critic)
            target_y = R_batch + GAMMA * Q_Spr_A * is_done_batch
            y = self.critic_net(S_batch, A_batch)

            # prediction loss for critic
            critic_loss = torch.sum(torch.pow(y - target_y, 2))

            # update critic network -> Q(S, A)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # find actor loss
            A_actor = self.actor_net(S_batch)
            actor_loss = -1 * torch.sum(self.critic_net(S_batch, A_actor))

            # update actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft_update target networks
            self.soft_update()

            n_steps += 1

        return critic_loss, actor_loss, R_total

    def soft_update(self):
        target_p = self.target_actor_net.state_dict()
        src_p = self.actor_net.state_dict()
        for key in src_p.keys():
            target_p[key] = target_p[key] * (1 - TAU) + src_p[key] * TAU
        self.target_actor_net.load_state_dict(target_p)

        target_p = self.target_critic_net.state_dict()
        src_p = self.critic_net.state_dict()
        for key in src_p.keys():
            target_p[key] = target_p[key] * (1 - TAU) + src_p[key] * TAU
        self.target_critic_net.load_state_dict(target_p)
