from pca import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import torch
from torch.autograd import Variable
from model import *
from environment import *

smooth = True

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
p1s, p2s = pcs(saved_time_series)
p1s = p1s[100:]
p2s = p2s[100:]
kernel_size = 10
kernel = np.ones(kernel_size) / kernel_size
p1s = np.convolve(p1s, kernel, mode='same')
p2s = np.convolve(p2s, kernel, mode='same')

fig, ax = plt.subplots()
xdata, ydata = [], []
p1data, p2data = [], []
ln, = plt.plot([], [], linestyle='-', marker='o')

components = pc_components(saved_time_series)

def init():
    ax.set(xlabel="PC2")
    ax.set(ylabel="PC1")
    ax.set_title("Regular Agent")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    return ln,

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
    if smooth:
        if frame < 10:
            xdata.append((sum(p1data[:]) + p1) / (len(xdata) + 1))
            ydata.append((sum(p2data[:] + p2) / len(ydata) + 1))
        else:
            xdata.append((sum(p1data[-9:]) + p1) / 10)
            ydata.append((sum(p2data[-9:]) + p2) / 10)
        p1data.append(p1)
        p2data.append(p2)
    else:
        xdata.append(p1)
        ydata.append(p2)
    ln.set_data(xdata[-10:], ydata[-10:])
    return ln,

ani = FuncAnimation(fig, update, frames=[i for i in range(1000)], init_func = init, blit=True, interval=1)
plt.show()
