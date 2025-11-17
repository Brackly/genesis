
from torchvision import transforms
from pathlib import Path
from PIL import Image
from src.factories.data_fetchers import Fetcher
from configs import config as cfg
import torch
from torch.utils.data import Subset
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = Path(root_path)
        self.transform = transform or transforms.Compose([
                transforms.Lambda(lambda img: img.convert('RGB')),
                transforms.Resize((224,224)),
                transforms.ToTensor()
             ]
        )

        self.images = []
        self.labels = []
        self.class_names = []

        for folder in self.root_path.iterdir():
            if folder.is_dir():
                class_idx = len(self.class_names)
                self.class_names.append(folder.name)

                for img_path in folder.glob("*.jp*g"):
                    self.images.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def get_data_loaders(
        config: cfg.DataLoaderConfig,
        fetcher: Fetcher,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
) -> dict:
    """
    Create stratified train, validation, and test dataloaders.
    Ensures each split maintains the same class distribution.
    """
    fetcher.fetch()

    # Create the full dataset
    full_dataset = Dataset(fetcher.local_save_path)

    # Get all labels for stratification
    labels = np.array(full_dataset.labels)
    indices = np.arange(len(labels))

    # First split: train+val vs test
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed
    )

    # Second split: train vs val
    train_val_labels = labels[train_val_indices]
    relative_val_size = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=relative_val_size,
        stratify=train_val_labels,
        random_state=seed
    )

    # Create subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Create dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        ),
        'val': torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        ),
        'test': torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
    }

    return dataloaders
