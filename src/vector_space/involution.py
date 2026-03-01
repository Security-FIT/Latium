import hydra
from omegaconf import DictConfig

import logging

from src.vector_space.common import involution
LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    involution(cfg)

if __name__ == "__main__":
    main()