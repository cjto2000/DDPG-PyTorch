import torch

import pickle
import os

from environment import Game
from replay_mem import ReplayMemory
from utilities import Orn_Uhlen
from ddpg import DDPG
from constants import *

# gam environment
env = Game()

# Create replay memory
memory = ReplayMemory(MAX_BUFF_SIZE, env.state_dim, env.n_actions)

agent = DDPG(env, memory)

def initialize_replay_mem():
    '''
    Initialize the replay memory.
    '''
    print("initializing replay memory...")
    S = env.reset()
    for _ in range(MAX_BUFF_SIZE):
        A = env.sample_action()
        S_prime, R, is_done = env.take_action(A)
        memory.add_to_memory((S, A, S_prime, R, is_done))
        if is_done:
            S = env.reset()
        else:
            S = S_prime


history = {
    "rewards" : [],
    "critic_loss" : [],
    "actor_loss": [],
}

if __name__ == "__main__":

    initialize_replay_mem()

    for i in range(N_EPISODES):
        noise = Orn_Uhlen(env.n_actions)
        l1, l2, R = agent.train_one_episode(BATCH_SIZE, noise)
        # TODO: maintain running rewards and losses
        if i % LOG_STEPS == 0:
            history["rewards"].append(R)
            history["critic_loss"].append(l1)
            history["actor_loss"].append(l2)
            print("Episode %5d -- Rewards : %.5f -- Losses: %.5f(a)  %.5f(c)" %(i, R, l2, l1))
        if i % SAVE_STEPS == 0:
            torch.save(agent.actor_net.state_dict(), actor_model_path)
            torch.save(agent.critic_net.state_dict(), critic_model_path)
    print("Training complete....")

