import torch
from torch.autograd import Variable
import gym
from model import *
from utilities import *

env = gym.make('BipedalWalker-v2')

state_dim = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
limit = float(env.action_space.high[0])

agent = Actor(state_dim, n_actions, limit)

try:
    agent.load_state_dict(torch.load("./saved_models/actor.pth"))
except:
    pass

is_done = False
S = env.reset()

while not is_done:
    S = Variable(torch.FloatTensor(S))
    A = agent(S).data.numpy()
    print(A)
    noise = Orn_Uhlen(n_actions)
    S_prime, R, is_done, _ = env.step(A + noise)
    S = S_prime
    env.render()
