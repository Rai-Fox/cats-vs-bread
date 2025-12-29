from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cats_vs_bread.configs import DataConfig


class CatsVsBreadDataModule(LightningDataModule):
    def __init__(self, data_config: DataConfig, batch_size: int):
        super().__init__()
        self.data_config = data_config
        self.batch_size = batch_size

        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage: str | None = None):
        self.train_dataset = datasets.ImageFolder(
            root=self.data_config.train_dir,
            transform=self.transform,
        )
        self.val_dataset = datasets.ImageFolder(
            root=self.data_config.val_dir,
            transform=self.transform,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
