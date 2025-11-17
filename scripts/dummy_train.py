import torch
from src.factories.datasets import dummy as dataset
from src.factories.models import dummy as models
from src.factories.trainers import vae_vanilla as trainer

import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)

    args = parser.parse_args()

    logger.info(f"Initializing experiment: {args.experiment_name}")

    dataloaders = dataset.get_data_loader()
    logger.info(f"{dataloaders.keys()}")

    # model initialization
    model = models.DummyModel()
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




