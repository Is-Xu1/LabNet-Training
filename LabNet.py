import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from obspy import read
from seisbench.models import VariableLengthPhaseNet

def apply_gaussian_mask(length, index, std=100):
    x = np.arange(length)
    return np.exp(-0.5 * ((x - index) / std) ** 2)