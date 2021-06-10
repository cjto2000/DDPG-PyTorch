from pca import demean, pcs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

saved_time_series = np.load("tmp/np_time_series.npy")
saved_time_series = demean(saved_time_series)
p1s, p2s = pcs(saved_time_series)
p1s = p1s[100:]
p2s = p2s[100:]
kernel_size = 10
kernel = np.ones(kernel_size) / kernel_size
p1s = np.convolve(p1s, kernel, mode='same')
p2s = np.convolve(p2s, kernel, mode='same')

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], linestyle='-', marker='o')

def init():
    ax.set(xlabel="PC2")
    ax.set(ylabel="PC1")
    ax.set_title("Regular Agent")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    return ln,

def update(frame):
    xdata.append(p1s[frame])
    ydata.append(p2s[frame])
    ln.set_data(xdata[-10:], ydata[-10:])
    return ln,

ani = FuncAnimation(fig, update, frames=[i for i in range(len(p1s))], init_func = init, blit=True, interval=100)
plt.show()

# for i in range(100, 200, 20):
#     p1 = p1s[i:i + 20]
#     p2 = p2s[i:i + 20]
#     df = pd.DataFrame({'x_axis': p1, 'y_axis': p2})
#     plt.plot(p1, p2, linestyle='-', marker='o')
#     plt.plot(p1[0], p1[0], 'og', linestyle='-', marker='o') # first el is green
#     plt.plot(p2[-1], p2[-1], 'ob', linestyle='-', marker='o') # last element is blue
#     # plt.plot('x_axis', 'y_axis', data=df, linestyle='-', marker='o')
#     # plt.plot(df['x_axis'][0], df['y_axis'][0], 'og')
#     # plt.plot(df['y_axis'][-1], df['y_axis'][-1], 'ob')
#     plt.show()
