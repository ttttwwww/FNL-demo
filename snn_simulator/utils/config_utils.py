"""Some tools for config usage"""
import copy
import importlib


def merge_config(default, user) -> dict:
    """
    To merge the default config and user config
    Args:
        default: Default config
        user: User config

    Returns:
        merged_config: Merged config
    """
    for key, value in user.items():
        if key in default and isinstance(default[key], dict) and isinstance(value, dict):
            merge_config(default[key], value)
        else:
            default[key] = value
    return copy.deepcopy(default)

def get_object_from_str(path_str:str):
    """
    Load object from string. To save module name in yaml file.
    """
    module_path, obj_name = path_str.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)
