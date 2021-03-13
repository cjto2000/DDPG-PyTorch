import torch
from torch.autograd import Variable
import gym
from model import *
from environment import *

env = Game()
agent = SuperLesionedActor(env.state_dim, env.n_actions, 16, env.limit)
actor_last_layer = ActorLastLayer(env.n_actions)
actor_last_layer.load_weights("models/damaged_models/actor.pth")

agent.load_state_dict(torch.load("./models/history01/actor.pth", map_location=torch.device('cpu')))

is_done = False
S = env.reset()

reward = 0
while not is_done:
    S = Variable(torch.FloatTensor(S))
    A, model_input = agent(S)
    A = A.data.numpy()
    real_A = actor_last_layer(model_input).data.numpy()
    S, R, is_done = env.take_action(real_A)
    reward += R
    env.render()
print(reward);
