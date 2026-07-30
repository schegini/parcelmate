import yaml
import copy


def get_cfg(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg
