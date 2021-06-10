from pca import demean, pcs
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

saved_time_series = np.load("tmp/np_time_series.npy")
saved_time_series = demean(saved_time_series)
p1s, p2s = pcs(saved_time_series)
sns.lineplot(x=[i for i in range(len(p1s))], y=p1s, label="PC1")
ax = sns.lineplot(x=[i for i in range(len(p2s))], y=p2s, label="PC2")
ax.set(xlabel="Timestep")
ax.set_title("Regular Agent")
plt.show()

saved_time_series = np.load("tmp/np_damaged_time_series.npy")
saved_time_series = demean(saved_time_series)
p1s, p2s = pcs(saved_time_series)
sns.lineplot(x=[i for i in range(len(p1s))], y=p1s, label="PC1")
ax = sns.lineplot(x=[i for i in range(len(p2s))], y=p2s, label="PC2")
ax.set(xlabel="Timestep")
ax.set_title("Damaged Agent")
plt.show()

saved_time_series = np.load("tmp/np_recovered_time_series.npy")
print(saved_time_series.shape)
saved_time_series = demean(saved_time_series)
p1s, p2s = pcs(saved_time_series)
sns.lineplot(x=[i for i in range(len(p1s))], y=p1s, label="PC1")
ax = sns.lineplot(x=[i for i in range(len(p2s))], y=p2s, label="PC2")
ax.set(xlabel="Timestep")
ax.set_title("Recovered Agent")
plt.show()