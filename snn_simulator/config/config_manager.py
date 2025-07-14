"""
Mange the parameters of simulation.Including
    hyperparameters for network
    parameters for training such as learning rate,batch size,optimizer criterion etc...
The config in python program will be saved a dict format.And will be saved in a yaml file.
The example file is shown in examole/config_example.yaml
"""
import os

from . import logger

import yaml




class ConfigManager:
    """
    Config interface.Save and load config from yaml file. Generate config dict for simulator.
    """

    def __init__(self, config_path):
        self.config_path = config_path
        self.config_dict = None

    def load_config(self):
        """
        Load config from yaml file.
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_dict = yaml.safe_load(f)
            logger.info(f'Loading config from {self.config_path}')
        except Exception as e:
            logger.error({e})
            logger.error(f'Failed to load config from {os.path.abspath(self.config_path)}')
        return self.config_dict

    def save_config(self, save_path):
        """
        Save config to yaml file.

        Args:
            save_path: The path to save config to.
        """
        try:
            yaml.dump(self.config_dict, save_path)
            logger.info(f'Saving config to {save_path}')
        except Exception as e:
            logger.error(f'Failed to save config to {save_path}: {e}')


