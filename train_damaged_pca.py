import torch

import json
import pickle
import os

from environment import Game
from replay_mem import ReplayMemoryTS
from ddpg_damaged_pca import *
from constants import *
from torch.autograd import Variable

start_over = True
hidden_dim = 16
include_loss = True

# gam environment
env = Game()

# Create replay memory
memory = ReplayMemoryTS(MAX_BUFF_SIZE, env.state_dim, env.n_actions)


# DDPG agent
agent = DDPG(env, memory, hidden_dim, include_loss=include_loss)

def initialize_replay_mem():
    '''
    Initialize the replay memory.
    '''
    print("Initializing replay memory...")
    S = env.reset()
    n = 0
    for _ in range(MAX_BUFF_SIZE):
        S_var = Variable(torch.FloatTensor(S)).to(device)
        A = env.sample_action()
        S_prime, R, is_done = env.take_action(A)
        memory.add_to_memory((S, A, S_prime, R, is_done, n))
        if is_done or n == 999:
            S = env.reset()
            n = 0
        else:
            S = S_prime
            n += 1


def write_history(path, hist):
  with open(path, "w") as f:
    json.dump(hist, f)


if os.path.exists("mem_init_pca.pkl") and start_over:
  memory = pickle.load(open("mem_init_pca.pkl", "rb"))
else:
  initialize_replay_mem()
  pickle.dump(memory, open("mem_init_pca.pkl", "wb"))

history = {
    "timesteps": [],
    "running_rewards" : [],
    "critic_loss": [],
    "actor_loss": [],
    "pca_loss": [],
    "rewards": [],
    "best_reward": 0,
    "total_timesteps": 0
}


if __name__ == "__main__":
    running_R = -100
    best_R = -float("inf")
    time_steps = 0
    reward = 0
    for i in range(1, N_EPISODES):
        l1, l2, R, n_steps, loss = agent.train_one_episode(BATCH_SIZE)
        time_steps += n_steps
        running_R = 0.9 * running_R + 0.1 * R
        reward += R
        # TODO: maintain running rewards and losses
        if i % LOG_STEPS == 0:
            history["timesteps"].append(time_steps)
            history["running_rewards"].append(round(running_R, 2))
            history["actor_loss"].append(round(l2.item(), 2))
            history["rewards"].append(round(reward / LOG_STEPS, 2))
            history["total_timesteps"] += time_steps
            history["pca_loss"].append(round(loss.item(), 2))
            time_steps = 0
            write_history("history.json", history)
            print("Episode %5d -- Running Rewards : %.5f -- Reward: %.5f -- Losses: %.5f(a) %.5f(c) %.5f(pca) -- Best Reward: %.5f -- Time steps: %d" %(i, running_R, reward / LOG_STEPS, l2, l1, loss, best_R, history["total_timesteps"]))
            reward = 0
        if R > best_R:
            best_R = R
            history["best_reward"] = round(best_R, 2)
            torch.save(agent.actor_net.state_dict(), actor_model_path)
            torch.save(agent.critic_net.state_dict(), critic_model_path)
        if R > 250:
            print("NOISE IS ZERO")
        # if i % SAVE_STEPS == 0:
        #     pickle.dump(memory, open("mem.pkl", "wb"))
        #     torch.save(agent.actor_net.state_dict(), actor_save_path)
        #     torch.save(agent.critic_net.state_dict(), critic_save_path)
        #     torch.save(agent.target_actor_net.state_dict(), target_actor_save_path)
        #     torch.save(agent.target_critic_net.state_dict(), target_critic_save_path)
    print("Training complete....")