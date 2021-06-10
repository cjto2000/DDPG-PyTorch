from pca import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import torch
from torch.autograd import Variable
from model import *
from environment import *

env = Game()
en_bool = True
agent = Actor(env.state_dim, env.n_actions, env.limit, en=en_bool)
agent.load_state_dict(torch.load("./models/history04/actor.pth", map_location=torch.device('cpu')))

is_done = False
env = Game()
S = env.reset()
env.env.seed(0)

saved_time_series = np.load("tmp/np_time_series.npy")
mean = np.mean(saved_time_series, axis=0)
fig, ax = plt.subplots()
xdata, ydata = [], []
x2data, y2data = [], []
ln1, = plt.plot([], [])
ln2, = plt.plot([], [])

components = pc_components(saved_time_series)

def init():
    ax.set(xlabel="Timestep")
    ax.set_title("Regular Agent")
    ax.set_xlim(0, 1000)
    ax.set_ylim(-50, 50)
    return ln1, ln2,


def update(frame):
    global S
    global agent
    S = Variable(torch.FloatTensor(S))
    A, en_input = agent(S)
    A = A.data.numpy()
    S, _, _ = env.take_action(A)
    env.render()
    if frame == 999:
        env.env.close()
        exit()
    data = np.expand_dims(en_input.data.numpy(), axis=0)
    data -= mean
    pc_values = components @ data.transpose(1, 0)
    p1 = pc_values[0]
    p2 = pc_values[1]
    xdata.append(frame)
    ydata.append(p1)
    x2data.append(frame)
    y2data.append(p2)
    ln1.set_data(xdata, ydata)
    ln2.set_data(x2data, y2data)
    return ln1, ln2,


ani = FuncAnimation(fig, update, frames=[_ for _ in range(1000)],
                    init_func=init, blit=True, interval=1)
plt.show()