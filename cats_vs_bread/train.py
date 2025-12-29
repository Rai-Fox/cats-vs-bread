from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger

from cats_vs_bread.configs import TrainConfig
from cats_vs_bread.models.data_module import CatsVsBreadDataModule
from cats_vs_bread.models.lightning_module import CatsVsBreadModel


def train_model(train_config: TrainConfig):
    data_module = CatsVsBreadDataModule(train_config.data, train_config.batch_size)
    lightning_model = CatsVsBreadModel(train_config)
    logger = MLFlowLogger(
        experiment_name=train_config.logger.experiment_name,
        tracking_uri=train_config.logger.tracking_uri,
    )

    trainer = Trainer(
        max_epochs=train_config.max_epochs,
        logger=logger,
    )
    trainer.fit(lightning_model, datamodule=data_module)

    
