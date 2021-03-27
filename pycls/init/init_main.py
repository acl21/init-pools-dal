import numpy as np 
import torch
from .sampling import SelfSupervisionSampling, ClusteringSampling
import pycls.utils.logging as lu
import os

logger = lu.get_logger(__name__)

class InitialPool:
    """
    Implements initial pool sampling methods.
    """

    def __init__(self, dataObj, cfg):
        self.dataObj = dataObj
        self.cfg = cfg

    def sample_from_uSet(self, trainDataset):
        """
        Sample from uSet using cfg.INIT_POOL.SAMPLING_FN.

        INPUT
        ------
        trainDataset: PyTorch dataset object (basically full data)

        OUTPUT
        -------
        Returns initSet, uSet
        """
        assert (self.cfg.INIT_POOL.INIT_RATIO > 0) & (self.cfg.INIT_POOL.INIT_RATIO < 1) , "Expected a label ration between 0 and 1"

        if self.cfg.INIT_POOL.SAMPLING_FN == 'rotation':
            initSet, uSet = SelfSupervisionSampling(trainDataset=trainDataset, ratio=self.cfg.INIT_POOL.INIT_RATIO)
        elif self.cfg.INIT_POOL.SAMPLING_FN == 'inpainting':
            initSet, uSet = SelfSupervisionSampling(trainDataset=trainDataset, ratio=self.cfg.INIT_POOL.INIT_RATIO)
        elif self.cfg.INIT_POOL.SAMPLING_FN == 'vae':
            initSet, uSet = SelfSupervisionSampling(trainDataset=trainDataset, ratio=self.cfg.INIT_POOL.INIT_RATIO)
        elif self.cfg.INIT_POOL.SAMPLING_FN == 'clustering':
            initSet, uSet = ClusteringSampling(trainDataset=trainDataset, ratio=self.cfg.INIT_POOL.INIT_RATIO)
        
        return initSet, uSet