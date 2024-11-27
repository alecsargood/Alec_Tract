import random
from monai.utils import set_determinism
import numpy as np
import torch

def setup_torch():

    random_seed = 42
    set_determinism(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device