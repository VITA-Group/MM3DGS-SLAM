import argparse
import os
import random

import numpy as np
import torch

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

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
    parser.add_argument("--config", "-c", type=str, help="Path to config file.")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--iteration", "-i", type=int, default=None)

    args = parser.parse_args()

    seed_everything()

    cfg = load_config(args.config)
    if args.iteration is not None:
        cfg["iteration"] = args.iteration
    if args.output is not None:
        cfg["outputdir"] = args.output
    slam = SLAM(cfg)
    psnr_list, ssim_list, lpips_list = slam.evaluate_images(args.iteration)

    print("  PSNR : {:>12.7f}".format(np.array(psnr_list).mean(), ".5"))
    print("  SSIM : {:>12.7f}".format(np.array(ssim_list).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(np.array(lpips_list).mean(), ".5"))

    print("Done.")
