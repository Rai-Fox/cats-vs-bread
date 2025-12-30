import random
from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cats_vs_bread.configs import DataConfig


class CatsVsBreadDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.root = root
        self.transform = transform
        self.image_paths = list(self.root.glob("*/*.jpeg"))
        random.shuffle(self.image_paths)
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(sorted({p.parent.name for p in self.image_paths}))
        }

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        image = self.transform(Image.open(image_path).convert("RGB"))
        label = self.class_to_idx[image_path.parent.name]
        return image, label  # type: ignore


class CatsVsBreadDataModule(LightningDataModule):
    def __init__(self, data_config: DataConfig) -> None:
        super().__init__()
        self.data_config = data_config

        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = CatsVsBreadDataset(root=self.data_config.train_dir, transform=self.transform)
        self.val_dataset = CatsVsBreadDataset(root=self.data_config.val_dir, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            persistent_workers=True,
        )
