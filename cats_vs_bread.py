#! /usr/bin/env python3

import fire
from omegaconf import OmegaConf

from cats_vs_bread.configs import CatsVsBreadConfig, compose_config
from cats_vs_bread.train import train_model
from cats_vs_bread.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CatsVsBreadCLI:
    def _compose_config(self, overrides: list[str]) -> CatsVsBreadConfig:
        config = compose_config(overrides=overrides)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")
        return config

    def train(self, *args: str) -> None:
        overrides = list(args)
        config = self._compose_config(overrides=overrides)
        train_model(config=config)


if __name__ == "__main__":
    fire.Fire(CatsVsBreadCLI)
