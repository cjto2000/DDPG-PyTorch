import torch
from torch.autograd import Variable
import gym
from model import *
from environment import *

env = Game()
# agent = DamagedActor(env.state_dim, env.n_actions, env.limit, en=True)
agent = Actor(env.state_dim, env.n_actions, env.limit, en=True)
en_model = EN(env.n_actions)
en_model.load_state_dict(torch.load("en_model/en.pth"))

try:
    agent.load_state_dict(torch.load("./models/good_models/actor.pth"))
except:
    print("No pretrained model found, using random model!!")
    pass

is_done = False
env.env.seed(0)
S = env.reset()
total_reward = 0
maxi = 0
while not is_done:
    S = Variable(torch.FloatTensor(S))
    A, en_input = agent(S)
    A = A.data.numpy()
    A_en = en_model(en_input)
    loss = torch.sum(torch.pow(torch.FloatTensor(A) - A_en, 2))
    if loss > maxi:
        maxi = loss
    A_en = A_en.data.numpy()
    S, R, is_done = env.take_action(A_en)
    total_reward += R
    env.render()
print(f"MAX: {loss}")
print(total_reward)