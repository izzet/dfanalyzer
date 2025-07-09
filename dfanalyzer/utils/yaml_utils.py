import yaml
from os.path import isfile
from typing import Callable


def load_persisted(path: str, fallback: Callable, persist=True):
    if isfile(path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    data = fallback()
    if persist:
        with open(path, 'w') as file:
            yaml.dump(data, file, sort_keys=True)
    return data
