#! /usr/bin/env python3

import fire

from cats_vs_bread.configs import CatsVsBreadConfig, compose_config
from cats_vs_bread.train import train_model


class CatsVsBreadCLI:
    def _compose_config(self) -> CatsVsBreadConfig:
        config = compose_config()
        return config

    def train(self) -> None:
        config = self._compose_config()
        train_model(train_config=config.train)


if __name__ == "__main__":
    fire.Fire(CatsVsBreadCLI)
