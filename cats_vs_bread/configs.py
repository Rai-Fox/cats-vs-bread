from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

DIR = Path(__file__).parent.parent
CONFIG_DIR = DIR / "configs"
CONFIG_NAME = "config"


@dataclass
class ModelConfig:
    model_name: str
    num_classes: int
    pretrained: bool
    output_path: Path


@dataclass
class DataConfig:
    train_dir: Path
    val_dir: Path


@dataclass
class MLFlowLoggerConfig:
    experiment_name: str
    tracking_uri: str


@dataclass
class TrainConfig:
    model: ModelConfig
    data: DataConfig
    logger: MLFlowLoggerConfig

    batch_size: int
    max_epochs: int
    learning_rate: float


@dataclass
class CatsVsBreadConfig:
    train: TrainConfig


cs = ConfigStore.instance()
cs.store(name=CONFIG_NAME, node=CatsVsBreadConfig)


def compose_config(
    overrides: list[str] | None = None,
    config_path: Path = CONFIG_DIR,
    config_name: str = CONFIG_NAME,
) -> CatsVsBreadConfig:
    with hydra.initialize(version_base=None, config_path=str(config_path)):
        hydra_config = hydra.compose(config_name=config_name, overrides=overrides)
        config = hydra.utils.instantiate(hydra_config, _recursive_=False)
        return config
