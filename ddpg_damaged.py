import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os
import numpy as np

from constants import *
from model import SuperLesionedActor, Critic, EN, ActorLastLayer

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    def __init__(self, env, memory, hidden_dim, include_loss=False):
        self.env = env

        n_inp = env.state_dim
        n_out = env.n_actions
        a_limit = env.limit
        self.include_loss = include_loss

        # self.en_net = EN(n_out).to(device)
        # self.en_net.load_state_dict(torch.load("damaged_models/en.pth"))
        # self.en_net.freeze_parameters()

        self.actor_net = SuperLesionedActor(n_inp, n_out, hidden_dim, a_limit).to(device)
        self.actor_net.load_weights(path="models/damaged_models/actor.pth") # load weights
        self.actor_net.load_en_weights(path="models/en_models/en_damaged.pth") # replace last layer with en network weights
        # self.actor_net.zero_weights() # zero weights of actor to damage it
        self.actor_net.freeze_parameters() # freeze all parameters except for the CPN
        self.critic_net = Critic(n_inp, n_out).to(device)
        self.critic_net.load_state_dict(torch.load("models/damaged_models/critic.pth"))

        self.actor_last_layer = ActorLastLayer(n_out).to(device) # model for getting actual action to be done
        self.actor_last_layer.load_weights(path="models/damaged_models/actor.pth")

        self.target_actor_net = SuperLesionedActor(n_inp, n_out, hidden_dim, a_limit).to(device)
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
            # self.env.render()
            S_var = Variable(torch.FloatTensor(S)).unsqueeze(0).to(device)
            en_output, model_input = self.actor_net(S_var)
            en_output = en_output.detach()
            # model_input = model_input.detach()
            # A_pred = self.actor_last_layer(model_input).detach() # get the actual action to be done
            noise = self.noise.sample()
            A = (en_output.data.cpu().numpy() + noise)[0][:] # use en_output instead of "actual" output
            S_prime, R, is_done = self.env.take_action(A)
            # store transition in replay memory (use surrogate action aka en output)
            self.memory.add_to_memory((S, A, S_prime, R, is_done))
            # update the next state for next iteration
            S = S_prime
            R_total += R

            # Training on samples drawn from replay memory
            S_batch, A_batch, S_prime_batch, R_batch, is_done_batch = self.memory.sample(batch_size)

            # cast into variables
            S_batch = Variable(torch.FloatTensor(S_batch)).to(device)
            A_batch = Variable(torch.FloatTensor(A_batch)).to(device)
            S_prime_batch = Variable(torch.FloatTensor(S_prime_batch)).to(device)
            R_batch = Variable(torch.FloatTensor(R_batch)).to(device)
            is_done_batch = Variable(torch.FloatTensor(is_done_batch)).to(device)

            # Use en output to update critic
            A_en_critic, model_input = self.target_actor_net(S_prime_batch)
            Q_Spr_A = self.target_critic_net(S_prime_batch, A_en_critic).detach()
            target_y = R_batch + GAMMA * Q_Spr_A * (1 - is_done_batch)
            y = self.critic_net(S_batch, A_batch)

            # prediction loss for critic
            critic_loss = torch.mean(torch.pow(y - target_y, 2))

            # update critic network -> Q(S, A)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # find actor loss, use en output
            A_en, model_input = self.actor_net(S_batch)
            # prediction loss for difference between surrogate and real action
            loss = None
            if self.include_loss:
                target_A_batch = self.actor_last_layer(model_input).detach()
                squared_difference = torch.pow(A_en - target_A_batch, 2)
                loss = torch.mean(torch.sum(squared_difference, dim=1))

            actor_loss = -1 * torch.mean(self.critic_net(S_batch, A_en)) + (loss if self.include_loss else 0)

            # update actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft update target networks
            self.soft_update()

            n_steps += 1

        self.noise.reset()
        return critic_loss, actor_loss, R_total, n_steps

    def soft_update(self):
      for target, src in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
          target.data.copy_(target.data * (1.0 - TAU) + src.data * TAU)

      for target, src in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
          target.data.copy_(target.data * (1.0 - TAU) + src.data * TAU)
