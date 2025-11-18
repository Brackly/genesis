import kagglehub
from pathlib import Path
import shutil
import random
from typing import Dict, List, Tuple
import json
from genesis.data_fetchers.base import Fetcher


class KaggleFetcher(Fetcher):
    def __init__(self, data_config):
        """
        Initialize KaggleFetcher with data configuration

        Args:
            data_config: Configuration object containing kaggle_dataset_path and kaggle_dataset_path_suffix
        """
        super().__init__()
        self.data_path = Path().cwd() / "data"
        self.config = data_config
        self.data_path_extension = data_config.kaggle_dataset_path_suffix

        # Split ratios
        self.train_ratio = 0.75
        self.val_ratio = 0.20
        self.test_ratio = 0.05

        # Random seed for reproducibility
        self.random_seed = 42

        # Supported image extensions
        self.image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

    def fetch(self, reorganize: bool = True, use_existing: bool = True, save_path:str |None = None):
        """
        Download dataset from Kaggle and optionally reorganize into train/val/test splits

        Args:
            reorganize: Whether to reorganize the dataset into train/val/test splits
            use_existing: Whether to use existing organized dataset if it exists
            save_path: Alternative path to local data folder
        """
        # Check if organized dataset already exists
        if use_existing:
            if self.data_path.exists():
                print(f"Using existing organized dataset at: {self.data_path.exists}")
                self.print_final_stats()
                return self.data_path
            elif save_path is not None and Path(save_path).exists():
                print(f"Using existing organized dataset at: {save_path}")
                save_path = Path(save_path)

        # Download dataset
        if save_path is None:
            save_path = kagglehub.dataset_download(self.config.kaggle_dataset_path)
            save_path = Path(save_path) / self.config.kaggle_dataset_path_suffix
            print(f"Dataset downloaded to: {self.data_path}")

        if reorganize:
            # Reorganize into train/val/test splits
            self.reorganize_dataset(save_path=save_path)

        return self.data_path


    def _get_image_paths(self, directory: Path) -> List[Path]:
        """Get all image paths from a directory"""
        images = []
        for ext in self.image_extensions:
            images.extend(directory.glob(ext))
        return images

    def analyze_dataset(self,save_path:Path) -> Tuple[Dict[str, List[Path]], int, Dict[str, int]]:
        """
        Analyze the dataset structure and count images per class

        save_path: Path to local data folder

        Returns:
            Tuple of (class_paths dict, minimum count, class_counts dict)
        """

        class_counts = {}
        class_paths = {}

        # Iterate through each subdirectory (assumed to be class folders)
        for class_dir in sorted(save_path.iterdir()):
            if class_dir.is_dir():
                # Get all images in this class directory
                images = self._get_image_paths(class_dir)

                if images:  # Only include classes with images
                    class_name = class_dir.name
                    class_counts[class_name] = len(images)
                    class_paths[class_name] = images

        if not class_counts:
            raise ValueError(f"No images found in {save_path}. "
                             f"Supported extensions: {self.image_extensions}")

        # Print analysis
        print("\n" + "=" * 50)
        print("DATASET ANALYSIS")
        print("=" * 50)
        print(f"Total classes: {len(class_counts)}")
        print(f"Total images: {sum(class_counts.values())}")
        print("\nImages per class:")

        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name:15s}: {count:4d} images")

        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        avg_count = sum(class_counts.values()) / len(class_counts)

        print(f"\nStatistics:")
        print(f"  Min images per class: {min_count}")
        print(f"  Max images per class: {max_count}")
        print(f"  Avg images per class: {avg_count:.1f}")
        print(f"  Imbalance ratio: {max_count / min_count:.2f}x")

        return class_paths, min_count, class_counts

    def reorganize_dataset(self, force_recreate: bool = False, save_path:str = None) -> None:
        """
        Reorganize dataset into train/val/test with uniform distribution

        Args:
            force_recreate: If True, recreate even if organized dataset exists
            save_path: Alternative path to local data folder
        """
        self.data_path.mkdir(parents=True, exist_ok=True)
        # Analyze current structure
        save_path = Path(save_path) if isinstance(save_path,str) else save_path

        class_paths, min_count, class_counts = self.analyze_dataset(save_path=save_path)

        # Calculate samples per split (using minimum for uniform distribution)
        samples_per_class = min_count
        train_per_class = int(samples_per_class * self.train_ratio)
        val_per_class = int(samples_per_class * self.val_ratio)
        test_per_class = samples_per_class - train_per_class - val_per_class

        # Calculate total data usage
        total_available = sum(class_counts.values())
        total_used = samples_per_class * len(class_counts)
        usage_percent = (total_used / total_available) * 100

        print("\n" + "=" * 50)
        print("SPLIT CONFIGURATION")
        print("=" * 50)
        print(f"Using uniform distribution with {samples_per_class} samples per class")
        print(f"Data utilization: {total_used}/{total_available} ({usage_percent:.1f}%)")
        print(f"\nSamples per class:")
        print(f"  Train: {train_per_class:4d} ({self.train_ratio * 100:.0f}%)")
        print(f"  Val:   {val_per_class:4d} ({self.val_ratio * 100:.0f}%)")
        print(f"  Test:  {test_per_class:4d} ({self.test_ratio * 100:.0f}%)")
        print(f"\nTotal samples per split:")
        print(f"  Train: {train_per_class * len(class_counts):5d}")
        print(f"  Val:   {val_per_class * len(class_counts):5d}")
        print(f"  Test:  {test_per_class * len(class_counts):5d}")

        # Create new directory structure
        organized_path = self.data_path

        print("\n" + "=" * 50)
        print("REORGANIZING DATASET")
        print("=" * 50)

        # Create split directories
        for split in ['train', 'val', 'test']:
            for class_name in class_paths.keys():
                (organized_path / split / class_name).mkdir(parents=True, exist_ok=True)

        # Set random seed for reproducibility
        random.seed(self.random_seed)

        # Copy files to new structure
        for class_idx, (class_name, image_paths) in enumerate(class_paths.items()):
            # Randomly sample and shuffle
            selected_images = random.sample(image_paths, samples_per_class)
            random.shuffle(selected_images)

            # Split into train/val/test
            train_images = selected_images[:train_per_class]
            val_images = selected_images[train_per_class:train_per_class + val_per_class]
            test_images = selected_images[train_per_class + val_per_class:]

            # Copy to respective directories
            splits_data = [
                ('train', train_images),
                ('val', val_images),
                ('test', test_images)
            ]

            for split_name, split_images in splits_data:
                for img_path in split_images:
                    dest = organized_path / split_name / class_name / img_path.name
                    shutil.copy2(img_path, dest)

            print(f"[{class_idx + 1}/{len(class_paths)}] {class_name:15s}: "
                  f"train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

        # Save split information for reproducibility
        split_info = {
            'original_path': str(save_path),
            'organized_path': str(organized_path),
            'random_seed': self.random_seed,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'samples_per_class': samples_per_class,
            'train_per_class': train_per_class,
            'val_per_class': val_per_class,
            'test_per_class': test_per_class,
            'classes': list(class_paths.keys()),
            'original_class_counts': class_counts
        }

        with open(organized_path / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)

        # Update the local_save_path to point to organized dataset
        self.data_path = organized_path

        print(f"\n✓ Dataset reorganized at: {self.data_path}")
        print(f"✓ Split info saved to: {organized_path / 'split_info.json'}")

        # Print final statistics
        self.print_final_stats()

    def print_final_stats(self):
        """Print statistics of the reorganized dataset"""
        print("\n" + "=" * 50)
        print("FINAL DATASET STRUCTURE")
        print("=" * 50)

        total_dataset_images = 0

        for split in ['train', 'val', 'test']:
            split_path = self.data_path / split
            if not split_path.exists():
                print(f"{split.capitalize()}: Not found")
                continue

            total_images = 0
            class_counts = {}

            for class_dir in sorted(split_path.iterdir()):
                if class_dir.is_dir():
                    count = len(self._get_image_paths(class_dir))
                    total_images += count
                    class_counts[class_dir.name] = count

            total_dataset_images += total_images

            # Check if uniform
            counts = list(class_counts.values())
            is_uniform = len(set(counts)) == 1 if counts else False

            print(f"\n{split.upper()}:")
            print(f"  Total images: {total_images}")
            print(f"  Classes: {len(class_counts)}")
            if counts:
                print(f"  Images per class: {counts[0]}")
                print(f"  Uniform distribution: {'✓' if is_uniform else '✗'}")

                if not is_uniform:
                    print(f"  Range: {min(counts)} - {max(counts)}")

        print(f"\nTotal dataset size: {total_dataset_images} images")

        # Load and display split info if it exists
        split_info_path = self.data_path / 'split_info.json'
        if split_info_path.exists():
            with open(split_info_path, 'r') as f:
                split_info = json.load(f)
                print(f"\nRandom seed used: {split_info['random_seed']}")
                print(f"Split ratios: {split_info['train_ratio']:.0%} / "
                      f"{split_info['val_ratio']:.0%} / {split_info['test_ratio']:.0%}")

    def get_data_paths(self) -> Dict[str, Path]:
        """
        Get paths to train, val, and test directories

        Returns:
            Dictionary with 'train', 'val', 'test' paths
        """
        if self.data_path is None:
            raise ValueError("Dataset not yet fetched. Call fetch() first.")

        paths = {}
        for split in ['train', 'val', 'test']:
            split_path = self.data_path / split
            if split_path.exists():
                paths[split] = split_path
            else:
                print(f"Warning: {split} directory not found at {split_path}")

        return paths