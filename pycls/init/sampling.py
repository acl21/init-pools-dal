import numpy as np 
import torch
from statistics import mean
import gc
import os
import math
import sys
import time
import pickle
import math
from copy import deepcopy
from tqdm import tqdm

from scipy.spatial import distance_matrix
import torch.nn as nn

class EntropyLoss(nn.Module):
    """
    This class contains the entropy function implemented.
    """
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x, applySoftMax=True):
        #Assuming x : [BatchSize, ]

        if applySoftMax:
            entropy = torch.nn.functional.softmax(x, dim=1)*torch.nn.functional.log_softmax(x, dim=1)
        else:
            entropy = x * torch.log2(x)
        entropy = -1*entropy.sum(dim=1)
        return entropy 

class SelfSupervisionSampling:
    """
    Here we implement different self-supervision based initial pool sampling methods.
    """

    def __init__(self, dataObj, cfg):
        self.cfg = cfg
        self.cuda_id = 0 if cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("ensemble") else torch.cuda.current_device()
        self.dataObj = dataObj

class ClusteringSampling:
    """
    Here we implement different the clustering based initial pool sampling methods.
    """