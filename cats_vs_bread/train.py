import dataclasses

import torch
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger

from cats_vs_bread.configs import CatsVsBreadConfig
from cats_vs_bread.models.data_module import CatsVsBreadDataModule
from cats_vs_bread.models.lightning_module import CatsVsBreadModel


def train_model(config: CatsVsBreadConfig) -> None:
    torch.set_float32_matmul_precision("medium")
    data_module = CatsVsBreadDataModule(data_config=config.data)
    lightning_model = CatsVsBreadModel(model_config=config.model, train_config=config.train)
    logger = MLFlowLogger(
        experiment_name=config.logger.experiment_name,
        tracking_uri=config.logger.tracking_uri,
    )
    logger.log_hyperparams(dataclasses.asdict(config))
    trainer = Trainer(
        max_epochs=config.train.max_epochs,
        logger=logger,
        enable_checkpointing=False,
    )
    trainer.fit(lightning_model, datamodule=data_module)
    trainer.save_checkpoint(config.model.output_path)
