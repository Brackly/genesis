import torch
from factories.datasets import DatasetFactory,DataLoaderFactory
from models import vanilla_vae as models
from data_fetchers import data_fetcher as fetcher
from configs import config
from trainers import neural_net as trainer

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
    parser.add_argument('--use_existing', type=bool, required=True,default=False)
    parser.add_argument('--batch_size', type=int, required=True, default=1000)
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()

    logger.info(f"Initializing experiment: {args.experiment_name}")

    # load configs
    data_config = config.DataConfig()
    data_loader_config = config.DataLoaderConfig()




    # data initialization
    print("Step 1: Fetching and organizing data...")
    fetcher = fetcher.KaggleFetcher(data_config)
    data_path = fetcher.fetch(reorganize=True, use_existing=args.use_existing)

    # Step 2: Create data_fetchers
    print("\nStep 2: Creating data_fetchers...")
    dataset_factory = DatasetFactory(
        data_path=data_path,
    )

    datasets_dict = dataset_factory.create_datasets()

    # Step 3: Create dataloaders
    print("\nStep 3: Creating dataloaders...")
    dataloader_factory = DataLoaderFactory(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    dataloaders = dataloader_factory.create_dataloaders(datasets_dict)

    # Get number of classes
    num_classes = dataset_factory.get_num_classes()

    print("\n" + "=" * 50)
    print("DATA PIPELINE SETUP COMPLETE")
    print("=" * 50)
    print(f"Number of classes: {num_classes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dataloaders created: {list(dataloaders.keys())}")

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

    trainer.train(epochs=args.epochs,validation_step=5, visualization_step=10)




