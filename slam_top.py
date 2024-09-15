# Top level module for running SLAM

import argparse
import random
import os
import numpy as np
import torch

from configs.config import load_config
from slam.SLAM import SLAM


def seed_everything(seed=0):
    """
    Set the `seed` value for torch and numpy seeds. Also turns on
    deterministic execution for cudnn.

    Parameters:
    - seed:     A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed} (type: {type(seed)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for running SLAM")
    parser.add_argument("--config", type=str, help="Path to config file.")

    args = parser.parse_args()

    seed_everything()

    cfg = load_config(args.config)
    slam = SLAM(cfg)
    slam.run()

    print("Done.")
