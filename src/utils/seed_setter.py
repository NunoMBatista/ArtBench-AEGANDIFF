import os
import random

import numpy as np
import torch


def set_global_seed(seed):
	# Keep all RNGs aligned so training/evaluation runs are reproducible.
	seed = int(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	# These flags trade some speed for deterministic CUDA convolution behavior.
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	return seed
