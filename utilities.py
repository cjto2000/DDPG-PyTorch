import torch
from torch.autograd import Variable

import numpy as np

def soft_update(target, source, TAU):
    for old, new in zip(target, source):
        old.data.copy_(old.data * (1 - TAU) + new.data * TAU)

def Orn_Uhlen(n_actions, mu=0, theta=0.15, sigma=0.2):
    X = mu * np.ones(n_actions)
    dX = theta * (mu - X)
    dX += sigma * np.random.randn(n_actions)
    return X + dX
