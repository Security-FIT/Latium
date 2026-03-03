import hydra
from omegaconf import DictConfig

import logging

from src.vector_space.common import involution
LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    acc = 0.0
    for loss in involution(cfg):
        acc += loss
    return acc

if __name__ == "__main__":
    main()