import yaml


def load_config(path):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
