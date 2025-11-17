from src.factories.trainers import BaseTrainer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from src.factories.datasets import plot_image
from collections.abc import Callable
import torch
from typing import Dict, Literal, Iterable, Any


class Trainer(BaseTrainer):
    def __init__(self,experiment_name:str,
                 model:torch.nn.Module,
                 loss_fn:torch.nn.Module,
                 optimizer:torch.optim.Optimizer,
                 dataloaders:Dict,#[Literal['train','val','test']:torch.utils.data.DataLoader],
                 scheduler:torch.optim.lr_scheduler= None):
        """
        A Trainer for a Model whose objective is reconstruction
        :param experiment_name: Name of the experiment
        :param model: Model to be trained
        :param loss_fn: Loss function
        :param optimizer: Optimizer to be used
        :param dataloaders: Dataloaders
        :param scheduler: Scheduler to be used

        """
        super().__init__()
        self.dataloaders : Dict[Literal['train','val','test']:torch.utils.data.DataLoader] = dataloaders
        self.model :  torch.nn.Module = model
        self.loss_fn : Callable = loss_fn
        self.optimizer : torch.optim = optimizer
        self.device : torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scheduler =  scheduler
        self.exp_dir = Path('experiments') / experiment_name
        self.writer = SummaryWriter(str(self.exp_dir / 'tensorboard'))
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []


    def score_fn(self,batch:Iterable):
        images,_ = batch
        images = images.to(self.device)
        reconstructed = self.model(images)
        # Calculate loss
        loss = self.loss_fn(reconstructed, images)
        return loss


    def train(self,epochs : int,
              score_fn:Callable[[Iterable[Any]],float|int] = None,
              save_fn:Callable[[int,str],None] = None,
              eval_fn:Callable[[int],None] = None,
              visualize_fn:Callable = None,
              visualize: bool = False,
              validation_step:int = 10,
              visualization_step:int = 10,):

        score_fn = self.score_fn if score_fn is None else score_fn
        save_fn = self.save if save_fn is None else save_fn
        eval_fn = self.eval if eval_fn is None else eval_fn
        if visualize:
            visualize_fn = self.visualize if visualize_fn is None else visualize_fn
        else:
            visualize_fn = None
        avg_val_loss = float('inf')
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_train_loss = 0

            progress_bar = tqdm(self.dataloaders['train'], desc=f'Epoch {epoch + 1}/{epochs}')
            for batch_idx, batch in enumerate(progress_bar):

                loss = score_fn(batch)
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item()
                })

                # Log to tensorboard
                global_step = epoch * len(self.dataloaders['train']) + batch_idx
                self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)

            # Calculate epoch averages
            avg_train_loss = epoch_train_loss / len(self.dataloaders['train'])

            self.train_losses.append(avg_train_loss)

            # Log epoch metrics
            self.writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Validation phase
            if (epoch + 1) % validation_step == 0:
                avg_val_loss = eval_fn(epoch)
                # Log validation metrics
                self.writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)

                # Learning rate scheduler step
                if self.scheduler is not None:
                    self.scheduler.step(avg_val_loss)

                # Save best model
                if avg_val_loss < self.best_val_loss:
                    best_val_loss = avg_val_loss
                    save_fn(epoch,"best_model.pt")
                    print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            # Visualization
            if (epoch + 1) % visualization_step == 0:
                if visualize_fn is not None:
                    visualize_fn(epoch)
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                save_fn(epoch,f"checkpoint_epoch_{epoch}.pt")

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            if (epoch + 1) % 10 == 0:
                print(f"Val Loss: {avg_val_loss:.4f}")
            print("-" * 50)

            # Save final model
        save_fn(epochs-1,"final_model.pt")

        # Close tensorboard writer
        self.writer.close()

        print(f"\nTraining complete! Results saved to {self.exp_dir}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return self.exp_dir

    def save(self,epoch,model_name:str):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, self.exp_dir / f'{model_name}.pt')

    def eval(self, epoch,score_fn:Callable):
        self.model.eval()
        epoch_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.dataloaders['val'], desc='Validation'):
                loss = score_fn(batch)
                epoch_val_loss += loss.item()

        # Calculate validation averages
        avg_val_loss = epoch_val_loss / len(self.dataloaders['val'])
        self.val_losses.append(avg_val_loss)
        return avg_val_loss

    def visualize(self, epoch):
        self.model.eval()
        with torch.no_grad():
            # Get a sample batch for visualization
            sample_batch = next(iter(self.dataloaders['val']))
            if isinstance(sample_batch, (list, tuple)):
                sample_image = sample_batch[0].to(self.device)
                sample_label = sample_batch[0].to(self.device) if len(sample_batch) > 1 else None
            else:
                sample_image = sample_batch[0].to(self.device)
                sample_label = None

            # Generate reconstructions
            if sample_label is not None:
                sample_recon = self.model(sample_image)
            else:
                sample_recon = self.model(sample_image)

            # Save visualization
            plot_image(sample_recon)

            # Log to tensorboard
            self.writer.add_images('Original', sample_image, epoch)
            self.writer.add_images('Reconstructed', sample_recon, epoch)


