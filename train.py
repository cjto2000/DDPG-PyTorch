import torch

import json
import pickle
import os

from environment import Game
from replay_mem import ReplayMemory
from ddpg import DDPG
from constants import *

# gam environment
env = Game()

# Create replay memory
memory = ReplayMemory(MAX_BUFF_SIZE, env.state_dim, env.n_actions)

# DDPG agent
agent = DDPG(env, memory)

def initialize_replay_mem():
    '''
    Initialize the replay memory.
    '''
    print("Initializing replay memory...")
    S = env.reset()
    for _ in range(MAX_BUFF_SIZE):
        A = env.sample_action()
        S_prime, R, is_done = env.take_action(A)
        memory.add_to_memory((S, A, S_prime, R, is_done))
        if is_done:
            S = env.reset()
        else:
            S = S_prime

def write_history(path, hist):
  with open(path, "w") as f:
    json.dump(hist, f)

if os.path.exists("mem.pkl"):
  memory = pickle.load(open("mem.pkl", "rb"))
else:
  initialize_replay_mem()
  pickle.dump(memory, open("mem.pkl", "wb"))

history = {
    "running_rewards" : [],
    "critic_loss" : [],
    "actor_loss": [],
    "rewards": [],
    "best_reward": 0
}

if __name__ == "__main__":
    running_R = -100
    best_R = -float("inf")
    for i in range(N_EPISODES):
        l1, l2, R = agent.train_one_episode(BATCH_SIZE)
        running_R = 0.9 * running_R + 0.1 * R
        # TODO: maintain running rewards and losses
        if i % LOG_STEPS == 0:
            history["running_rewards"].append(round(running_R, 2))
            history["critic_loss"].append(round(l1.item(), 2))
            history["actor_loss"].append(round(l2.item(), 2))
            history["rewards"].append(round(R, 2))
            write_history("history.json", history)
            print("Episode %5d -- Rewards : %.5f -- Losses: %.5f(a)  %.5f(c) -- Best Reward: %.5f" %(i, running_R, l2, l1, best_R))
        if R > best_R:
            best_R = R
            history["best_reward"] = round(best_R, 2)
            torch.save(agent.actor_net.state_dict(), actor_model_path)
            torch.save(agent.critic_net.state_dict(), critic_model_path)
    print("Training complete....")
