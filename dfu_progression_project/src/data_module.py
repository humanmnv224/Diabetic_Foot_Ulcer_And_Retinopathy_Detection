import lightning as L
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2 as T

from .dataset import DFUDataset


class DFUDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_csv_path: str,
        val_csv_path: str,
        image_size: int = 160,
        batch_size: int = 32,
        num_workers: int = 4,
        use_class_weighting: bool = True,
        use_weighted_sampler: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        if use_class_weighting and use_weighted_sampler:
            raise ValueError(
                "use_class_weighting and use_weighted_sampler cannot both be True"
            )

        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.use_class_weighting = use_class_weighting
        self.use_weighted_sampler = use_weighted_sampler

        # Define transformations
        self.train_transform = T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.RandomRotation(degrees=15),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.val_transform = T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def setup(self, stage=None):
        """Set up datasets for training and validation."""
        self.train_dataset = DFUDataset(
            self.train_csv_path, transform=self.train_transform
        )
        self.val_dataset = DFUDataset(self.val_csv_path, transform=self.val_transform)

        labels = self.train_dataset.labels.numpy()
        self.num_classes = len(np.unique(labels))
        self.class_weights = (
            self._compute_class_weights(labels) if self.use_class_weighting else None
        )

    def train_dataloader(self):
        """Returns a DataLoader for training data."""
        if self.use_weighted_sampler:
            sampler = self._get_weighted_sampler(self.train_dataset.labels.numpy())
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def _compute_class_weights(self, labels):
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(labels), y=labels
        )
        return torch.tensor(class_weights, dtype=torch.float32)

    def _get_weighted_sampler(self, labels: np.ndarray) -> WeightedRandomSampler:
        class_sample_count = np.array(
            [len(np.where(labels == label)[0]) for label in np.unique(labels)]
        )
        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[label] for label in labels])
        samples_weight = torch.from_numpy(samples_weight)
        return WeightedRandomSampler(
            weights=samples_weight, num_samples=len(labels), replacement=True
        )
