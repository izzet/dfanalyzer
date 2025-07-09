import hashlib
import os
from pathlib import Path


def dir_hash(directory: str):
    md5 = hashlib.md5()
    for root, dirs, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            modify_time = os.path.getmtime(path)
            md5.update(f"{path}{modify_time}".encode())
    return md5.hexdigest()


def ensure_dir(directory: str):
    Path(directory).mkdir(parents=True, exist_ok=True)
