from pca import demean, pcs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

saved_time_series = np.load("tmp/np_time_series.npy")
saved_time_series = demean(saved_time_series)
p1s, p2s = pcs(saved_time_series)
fig, ax = plt.subplots()
xdata, ydata = [], []
x2data, y2data = [], []
ln, = plt.plot([], [])
pc2, = plt.plot([], [])
frames = np.array([(i, p1s[i], p2s[i]) for i in range(len(p1s))])

def init():
    ax.set(xlabel="Timestep")
    ax.set_title("Regular Agent")
    ax.set_xlim(min([j[0] for j in frames]), max([j[0] for j in frames]))
    ax.set_ylim(min(j[1] for j in frames), max([j[1] for j in frames]))
    return ln, pc2,


def update(frame):
    xdata.append(frame)
    ydata.append(p1s[frame])
    x2data.append(frame)
    y2data.append(p2s[frame])
    ln.set_data(xdata, ydata)
    pc2.set_data(x2data, y2data)
    return ln, pc2,


ani = FuncAnimation(fig, update, frames=[i for i in range(1000)],
                    init_func=init, blit=True, interval=10)
plt.show()

""""
# Example using FuncAnimation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
plt.show()
"""
