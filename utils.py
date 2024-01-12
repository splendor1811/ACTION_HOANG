from models.model_single import Recognizer3D
from datasets.dataset import PoseC3DDataset
from models.losses import CrossEntrophyLoss, binary_cross_entrophy_with_logits
from models.optimizers import Optimizer
from models.schedulers import Schedulers
import argparse
import yaml
import os
import numpy as np
import random
import csv
import glob
import json
import shutil
import traceback
from collections import OrderedDict
import pickle
import time

#torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import gc

from tensorboardX import SummaryWriter
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
import wandb
import torch.nn.functional as F

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def one_hot(x, class_count = 9):
    return torch.eye(class_count)[x,:]

def label_smoothing(x, epsilon, class_count=9):
    one_hot_label = one_hot(x, class_count)
    one_hut_label_smooth = one_hot_label * (1 - epsilon) + (1 - one_hot_label) * epsilon / (class_count - 1)
    return one_hut_label_smooth
