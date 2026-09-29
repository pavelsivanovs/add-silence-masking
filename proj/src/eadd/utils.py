import logging

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def load_config(config_path: str):
    with open(config_path, "r") as config_file:
        config = OmegaConf.load(config_file)
        logger.info(f"Config read from {config_path}:\n{OmegaConf.to_yaml(config)}")
        return config
