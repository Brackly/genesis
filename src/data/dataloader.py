import torch
import torch.utils.data.dataloader as dl
from configs import config as cfg
from src.data.fetch import KaggleFetcher
from src.data import dataset as ds

def animal_data_loader(
        config:cfg.DataLoaderConfig=cfg.DataLoaderConfig(),
):
    fetcher = KaggleFetcher(data_config=cfg.DataConfig)
    fetcher.fetch()
    def dataset():
        return ds.AnimalDataset(fetcher.local_save_path)

    return dl.DataLoader(
        dataset=dataset(),
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,

    )