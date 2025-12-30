from typing import Any

import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

from cats_vs_bread.configs import ModelConfig, TrainConfig
from cats_vs_bread.models.model import CatsVsBreadClassfier


class CatsVsBreadModel(LightningModule):
    def __init__(self, model_config: ModelConfig, train_config: TrainConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.model = CatsVsBreadClassfier(model_config=self.model_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _calc_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        f1_score = torchmetrics.functional.f1_score(
            preds, labels, num_classes=self.model_config.num_classes, average="macro", task="binary"
        )
        roc_auc = torchmetrics.functional.auroc(F.softmax(logits, dim=1)[:, 1], labels, task="binary")
        return {
            "accuracy": accuracy.item(),
            "f1_score": f1_score.item(),
            "roc_auc": roc_auc.item(),  # type: ignore
        }

    def _step(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        return logits, loss

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:  # noqa: ANN401
        images, labels = batch
        _, loss = self._step(images, labels)
        self.log("train/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:  # noqa: ANN401
        images, labels = batch
        logits, loss = self._step(images, labels)
        metrics = self._calc_metrics(logits, labels)
        self.log("val/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric_name, metric_value in metrics.items():
            self.log(f"val/{metric_name}", metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_config.learning_rate)

        return optimizer
