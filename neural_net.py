import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
import gym
import random
import matplotlib.pyplot as plt
from itertools import count

class PolicyNet(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super(PolicyNet,self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ELU(),
            nn.Linear(32,act_dim)
        )

    def forward(self,obs:tensor) -> tensor :
        logits = self.net(obs)
        action_probs = self.softmax(logits)
        return action_probs


