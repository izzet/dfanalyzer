import os


def get_bool_env_var(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).lower() in ["true", "1", "yes"]
