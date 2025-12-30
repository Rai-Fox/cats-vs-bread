from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

DIR = Path(__file__).parent.parent
CONFIG_DIR = "../configs"
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
    num_workers: int
    batch_size: int


@dataclass
class MLFlowLoggerConfig:
    experiment_name: str
    tracking_uri: str


@dataclass
class TrainConfig:
    max_epochs: int
    learning_rate: float


@dataclass
class CatsVsBreadConfig:
    train: TrainConfig
    model: ModelConfig
    data: DataConfig
    logger: MLFlowLoggerConfig


cs = ConfigStore.instance()
cs.store(name=f"{CONFIG_NAME}_schema", node=CatsVsBreadConfig)


def to_dataclass(cfg: DictConfig, schema_type: type[CatsVsBreadConfig]) -> CatsVsBreadConfig:
    schema_cfg = OmegaConf.structured(schema_type)
    merged = OmegaConf.merge(schema_cfg, cfg)
    obj = OmegaConf.to_object(merged)
    return obj  # type: ignore


def compose_config(
    overrides: list[str] | None = None,
    config_path: str = CONFIG_DIR,
    config_name: str = CONFIG_NAME,
) -> CatsVsBreadConfig:
    with hydra.initialize(version_base=None, config_path=config_path):
        hydra_config = hydra.compose(config_name=config_name, overrides=overrides)
        config = to_dataclass(hydra_config, CatsVsBreadConfig)
        return config
