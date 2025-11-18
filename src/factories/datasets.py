from pathlib import Path
from typing import Dict, Optional, Tuple, Callable
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(DummyDataset, self).__init__()
        self.x = torch.randn(1000, 28)
        self.y = torch.randn(1000)

    def __len__(self):
        return 1000
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_data_loader():
    return {
        'train':torch.utils.data.DataLoader(
        dataset=DummyDataset(),
        batch_size=100,
        shuffle=True),
        'val':torch.utils.data.DataLoader(
        dataset=DummyDataset(),
            batch_size=100,
            shuffle=True),
        'test':torch.utils.data.DataLoader(
            dataset=DummyDataset(),
            batch_size=100,
        shuffle = True),
    }

class DatasetFactory:
    """Factory for creating data_fetchers with proper transforms for train/val/test splits"""

    def __init__(self, data_path: Path, image_size: int = 224, num_channels: int = 3):
        """
        Initialize the dataset factory

        Args:
            data_path: Path to the organized data directory (containing train/val/test folders)
            image_size: Size to resize images to
            num_channels: Number of channels (3 for RGB, 1 for grayscale)
        """
        self.data_path = data_path
        self.image_size = image_size
        self.num_channels = num_channels
        self.base_transforms = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]

        # Define transforms for each split
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
        self.test_transform = self._get_test_transform()

    def _get_train_transform(self) -> transforms.Compose:
        """Get training transforms with data augmentation"""
        transform_list = self.base_transforms

        #     [
        #     transforms.Resize((self.image_size, self.image_size)),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomRotation(degrees=10),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #     transforms.ToTensor(),
        # ]

        # Add grayscale conversion if single channel
        if self.num_channels == 1:
            transform_list.insert(0, transforms.Grayscale(num_output_channels=1))

        # Normalize based on number of channels
        if self.num_channels == 1:
            # Grayscale normalization
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        else:
            # RGB ImageNet normalization
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )

        return transforms.Compose(transform_list)

    def _get_val_transform(self) -> transforms.Compose:
        """Get validation transforms without augmentation"""
        transform_list = self.base_transforms

        # Add grayscale conversion if single channel
        if self.num_channels == 1:
            transform_list.insert(0, transforms.Grayscale(num_output_channels=1))

        # Normalize
        if self.num_channels == 1:
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        else:
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )

        return transforms.Compose(transform_list)

    def _get_test_transform(self) -> transforms.Compose:
        """Get test transforms (same as validation)"""
        return self._get_val_transform()

    def create_datasets(self,
                        custom_train_transform: Optional[Callable] = None,
                        custom_val_transform: Optional[Callable] = None,
                        custom_test_transform: Optional[Callable] = None) -> Dict[str, datasets.ImageFolder]:
        """
        Create data_fetchers for all splits

        Args:
            custom_train_transform: Optional custom transform for training
            custom_val_transform: Optional custom transform for validation
            custom_test_transform: Optional custom transform for testing

        Returns:
            Dictionary with 'train', 'val', 'test' data_fetchers
        """
        datasets_dict = {}

        # Use custom transforms if provided, otherwise use defaults
        train_transform = custom_train_transform or self.train_transform
        val_transform = custom_val_transform or self.val_transform
        test_transform = custom_test_transform or self.test_transform

        # Create data_fetchers for each split
        for split, transform in [
            ('train', train_transform),
            ('val', val_transform),
            ('test', test_transform)
        ]:
            split_path = self.data_path / split
            if split_path.exists():
                dataset = datasets.ImageFolder(
                    root=split_path,
                    transform=transform
                )
                datasets_dict[split] = dataset
                print(f"Created {split} dataset with {len(dataset)} samples from {split_path}")
            else:
                print(f"Warning: {split} directory not found at {split_path}")

        # Print class information from train dataset
        if 'train' in datasets_dict:
            print(f"\nNumber of classes: {len(datasets_dict['train'].classes)}")
            print(f"Classes: {datasets_dict['train'].classes}")

        return datasets_dict

    def get_num_classes(self) -> int:
        """Get the number of classes in the dataset"""
        train_path = self.data_path / 'train'
        if train_path.exists():
            return len([d for d in train_path.iterdir() if d.is_dir()])
        return 0


class DataLoaderFactory:
    """Factory for creating dataloaders from data_fetchers"""

    def __init__(self,
                 batch_size: int = 500,
                 num_workers: int = 0,
                 pin_memory: bool = True if torch.cuda.is_available() else False,
                 drop_last: bool = False):
        """
        Initialize the dataloader factory

        Args:
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.drop_last = drop_last

    def create_dataloaders(self,
                           datasets_dict: Dict[str, datasets.ImageFolder],
                           train_batch_size: Optional[int] = None,
                           val_batch_size: Optional[int] = None,
                           test_batch_size: Optional[int] = None) -> Dict[str, DataLoader]:
        """
        Create dataloaders from data_fetchers

        Args:
            datasets_dict: Dictionary with 'train', 'val', 'test' data_fetchers
            train_batch_size: Optional custom batch size for training
            val_batch_size: Optional custom batch size for validation
            test_batch_size: Optional custom batch size for testing

        Returns:
            Dictionary with 'train', 'val', 'test' dataloaders
        """
        dataloaders = {}

        # Define batch sizes
        batch_sizes = {
            'train': train_batch_size or self.batch_size,
            'val': val_batch_size or self.batch_size,
            'test': test_batch_size or self.batch_size
        }

        for split, dataset in datasets_dict.items():
            shuffle = (split == 'train')  # Only shuffle training data

            dataloader = DataLoader(
                dataset,
                batch_size=batch_sizes[split],
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last if split == 'train' else False
            )

            dataloaders[split] = dataloader

            print(f"Created {split} dataloader: "
                  f"batch_size={batch_sizes[split]}, "
                  f"num_batches={len(dataloader)}, "
                  f"shuffle={shuffle}")

        return dataloaders