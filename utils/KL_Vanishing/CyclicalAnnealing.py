import numpy as np
import math
import torch
from torch.nn import functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L


def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]

    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L


#  function  = 1 − cos(a), where a scans from 0 to pi/2
def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]

    # transform into [0, pi] for plots:

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
            v += step
            i += 1
    return L


def frange(start, stop, step, n_epoch):
    L = np.ones(n_epoch)
    v , i = start , 0
    while v <= stop:
        L[i] = v
        v += step
        i += 1
    return L


class Scaler():
    def __init__(self, input_shape, tau=0.5,  **kwargs):
        super(Scaler, self).__init__(**kwargs)

        self.tau = tau
        self.scale = torch.zeros(input_shape).to(DEVICE)

    def forward(self, inputs, mode='positive'):
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * F.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * F.sigmoid(-self.scale)

        return inputs * torch.sqrt(scale)
