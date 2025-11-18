import torch
from genesis.models import dummy_model as model
from genesis.factories.datasets import datasets as dataset
from genesis.factories import trainers as trainer

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
    model = model.DummyModel()
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

    trainer.run(epochs=args.epochs)




