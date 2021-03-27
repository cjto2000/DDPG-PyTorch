import torch
from torch.autograd import Variable
import gym
from model import *
from environment import *

env = Game()
agent = DamagedActor(env.state_dim, env.n_actions, env.limit, hidden_dim=16)
# agent = Actor(env.state_dim, env.n_actions, env.limit)
agent.load_state_dict(torch.load("./models/history08/actor.pth", map_location=torch.device('cpu')))
# agent.load_state_dict(torch.load("./models/history04/actor.pth", map_location=torch.device('cpu')))

is_done = False
env.env.seed(0)
S = env.reset()
n = 0
reward = 0
while not is_done and n < 1000:
    S = Variable(torch.FloatTensor(S))
    A = agent(S)
    A = A.data.numpy()
    S, R, is_done = env.take_action(A)
    env.render()
    reward += R
    n += 1

print(reward)