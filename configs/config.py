from dataclasses import dataclass

@dataclass(frozen=True)
class DataConfig:
    kaggle_dataset_path : str = 'alessiocorrado99/animals10'
    kaggle_dataset_path_suffix: str = 'raw-img'

@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 1000
    shuffle: bool  = True
    num_workers: int  = 0
    pin_memory: bool = False
    prefetch_factor: int = 0
    persistent_workers: bool = False
    pin_memory_device: str = ""
    train : float = 0.8
    val : float = 0.2