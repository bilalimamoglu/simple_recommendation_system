# app/utils.py

import numpy as np
import random

def set_seed(seed=42):
    """
    Sets the seed for reproducibility.

    Parameters:
    - seed: The seed value to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    # Add other libraries' seed settings if used
