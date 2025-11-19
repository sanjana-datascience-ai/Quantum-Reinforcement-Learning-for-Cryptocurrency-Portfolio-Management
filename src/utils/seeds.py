"""
Deterministic seeding helper.
"""

import random
import numpy as np
import os


def set_global_seed(seed: int = 42):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # Deterministic CuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # torch not installed or error setting determinism
        pass

    print(f"[seed] Global seed set to {seed}")
    return seed
