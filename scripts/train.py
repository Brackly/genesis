import torch
from src.factories.data_fetchers import kaggle_fetcher as fetcher
from src.factories.datasets import animal_10 as dataset
from src.factories.models import vanilla_vae as models
from src.factories.trainers import vae_vanilla as trainer
from configs import config

import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--train_ratio', type=float, required=True)
    parser.add_argument('--val_ratio', type=float, required=True)
    parser.add_argument('--test_ratio', type=float, required=True)

    args = parser.parse_args()

    logger.info(f"Initializing experiment: {args.experiment_name}")

    # load configs
    data_config = config.DataConfig()
    data_loader_config = config.DataLoaderConfig()

    # data initialization
    fetcher = fetcher.KaggleFetcher(data_config=data_config)
    dataloaders = dataset.get_data_loaders(
        config=data_loader_config,
        fetcher=fetcher,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    logger.info(f"{dataloaders.keys()}")

    # model initialization
    model = models.VanillaVAE()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = None
    loss_fn = torch.nn.MSELoss()

    # training initialization
    logger.info(f"Initializing training..")
    trainer = trainer.Trainer(experiment_name=args.experiment_name,
                              model=model,
                              optimizer=optimizer,
                              dataloaders=dataloaders,
                              scheduler=scheduler,
                              loss_fn=loss_fn
                              )

    trainer.train(epochs=args.epochs)




