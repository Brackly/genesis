from src.factories.trainers import BaseTrainer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from src.factories.datasets import plot_image
from collections.abc import Callable
import torch
import torch.nn as nn
from typing import Dict, Literal, Iterable, Any
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


class Trainer(BaseTrainer):
    def __init__(self, experiment_name: str,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 dataloaders: Dict,
                 scheduler: torch.optim.lr_scheduler = None,
                 num_classes: int = 10):
        """
        A Trainer for a Model whose objective is reconstruction
        :param experiment_name: Name of the experiment
        :param model: Model to be trained
        :param loss_fn: Loss function
        :param optimizer: Optimizer to be used
        :param dataloaders: Dataloaders
        :param scheduler: Scheduler to be used
        :param num_classes: Number of classes for per-class visualization
        """
        super().__init__()
        self.dataloaders: Dict[Literal['train', 'val', 'test']: torch.utils.data.DataLoader] = dataloaders
        self.model: torch.nn.Module = model
        self.loss_fn: Callable = loss_fn
        self.optimizer: torch.optim = optimizer
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scheduler = scheduler
        self.exp_dir = Path('experiments') / experiment_name
        self.writer = SummaryWriter(str(self.exp_dir / 'tensorboard'))
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.val_epochs = []  # Track actual validation epochs
        self.num_classes = num_classes

        # Store class samples for consistent visualization
        self.class_samples = self._get_class_samples()

        # Track metrics history
        self.best_model_epochs = []
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'learning_rate': [],
            'gradient_norm': []
        }

        # Log model architecture
        self._log_model_architecture()

    def _get_class_samples(self):
        """Get one sample from each class for consistent visualization"""
        class_samples = {}
        self.model.eval()

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images, labels = batch
                for i in range(len(labels)):
                    label = labels[i].item()
                    if label not in class_samples:
                        class_samples[label] = images[i:i + 1].to(self.device)
                    if len(class_samples) == self.num_classes:
                        return class_samples

        # If not enough classes found in validation, try train
        if len(class_samples) < self.num_classes:
            for batch in self.dataloaders['train']:
                images, labels = batch
                for i in range(len(labels)):
                    label = labels[i].item()
                    if label not in class_samples:
                        class_samples[label] = images[i:i + 1].to(self.device)
                    if len(class_samples) == self.num_classes:
                        return class_samples

        return class_samples

    def _log_model_architecture(self):
        """Log model architecture to tensorboard"""
        try:
            # Get a sample input
            sample_batch = next(iter(self.dataloaders['train']))
            sample_input = sample_batch[0][:1].to(self.device)

            # Add graph
            self.writer.add_graph(self.model, sample_input)

            # Log model parameters count
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            self.writer.add_text('Model/Info',
                                 f'Total Parameters: {total_params:,}\n'
                                 f'Trainable Parameters: {trainable_params:,}')
        except Exception as e:
            print(f"Could not log model architecture: {e}")

    def score_fn(self, batch: Iterable):
        images, _ = batch
        images = images.to(self.device)
        reconstructed = self.model(images)
        # Calculate loss
        loss = self.loss_fn(reconstructed, images)
        return loss

    def train(self, epochs: int,
              score_fn: Callable = None,
              save_fn: Callable = None,
              eval_fn: Callable = None,
              visualize_fn: Callable = None,
              visualize: bool = True,
              validation_step: int = 10,
              visualization_step: int = 10, ):

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
            batch_losses = []
            gradient_norms = []

            progress_bar = tqdm(self.dataloaders['train'], desc=f'Epoch {epoch + 1}/{epochs}')
            for batch_idx, batch in enumerate(progress_bar):
                loss = score_fn(batch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Calculate gradient norm before optimizer step
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                gradient_norms.append(total_norm)

                self.optimizer.step()

                batch_losses.append(loss.item())
                epoch_train_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'grad_norm': f'{total_norm:.3f}'
                })

                # Log batch metrics
                global_step = epoch * len(self.dataloaders['train']) + batch_idx
                self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)

            # Calculate epoch averages
            avg_train_loss = epoch_train_loss / len(self.dataloaders['train'])
            avg_grad_norm = np.mean(gradient_norms)

            self.train_losses.append(avg_train_loss)
            self.metrics_history['train_loss'].append(avg_train_loss)
            self.metrics_history['gradient_norm'].append(avg_grad_norm)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.metrics_history['learning_rate'].append(current_lr)

            # Log combined loss plot
            self._log_combined_losses(epoch)

            # Log training metrics
            self.writer.add_scalar('Metrics/gradient_norm', avg_grad_norm, epoch)
            self.writer.add_scalar('Metrics/learning_rate', current_lr, epoch)

            # Always log train loss to scalar for multi-line plot
            self.writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

            # Log loss distribution
            self.writer.add_histogram('Loss/train_distribution', np.array(batch_losses), epoch)

            # Validation phase
            if (epoch + 1) % validation_step == 0:
                avg_val_loss = eval_fn(epoch, score_fn)
                self.val_epochs.append(epoch)  # Track actual validation epochs
                self.metrics_history['val_loss'].append(avg_val_loss)

                # Log validation loss to scalar for multi-line plot
                self.writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)

                # Update combined plot after validation
                self._log_combined_losses(epoch)

                # Learning rate scheduler step
                if self.scheduler is not None:
                    self.scheduler.step(avg_val_loss)

                # Check for best model
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.best_model_epochs.append(epoch)

                    # Test on best model
                    test_loss = self._evaluate_test(epoch, score_fn)
                    self.metrics_history['test_loss'].append(test_loss)

                    # Visualize class reconstructions for best model
                    self._visualize_class_reconstructions(epoch, avg_val_loss, test_loss)

                    # Log model weights histograms
                    self._log_model_weights(epoch)

                    save_fn(epoch, "best_model.pt")
                    print(f"Saved best model with validation loss: {self.best_val_loss:.4f}")

                    # Log best model info
                    self._log_best_model_info(epoch, avg_val_loss, test_loss)

            # Visualization
            if (epoch + 1) % visualization_step == 0:
                if visualize_fn is not None:
                    visualize_fn(epoch)

            # Save checkpoint
            if (epoch + 1) % 50 == 0:
                save_fn(epoch, f"checkpoint_epoch_{epoch}.pt")

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            if (epoch + 1) % validation_step == 0:
                print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Gradient Norm: {avg_grad_norm:.3f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print("-" * 50)

        # Save final model
        save_fn(epochs - 1, "final_model.pt")

        # Log final summary
        self._log_training_summary()

        # Close tensorboard writer
        self.writer.close()

        print(f"\nTraining complete! Results saved to {self.exp_dir}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return self.exp_dir

    def _log_combined_losses(self, epoch):
        """Log train, val, and test losses on the same plot"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot training loss
        if self.train_losses:
            ax.plot(range(len(self.train_losses)), self.train_losses,
                    label='Train Loss', color='blue', linewidth=2, alpha=0.8)

        # Plot validation loss at actual validation epochs
        if self.val_losses and self.val_epochs:
            ax.plot(self.val_epochs, self.val_losses,
                    label='Val Loss', color='orange', marker='o',
                    markersize=6, linewidth=2, alpha=0.8)

            # Connect validation points with a line for better visibility
            if len(self.val_epochs) > 1:
                ax.plot(self.val_epochs, self.val_losses,
                        color='orange', linewidth=1, alpha=0.4, linestyle='--')

        # Plot test loss at best model points
        if self.test_losses and self.best_model_epochs:
            ax.scatter(self.best_model_epochs, self.test_losses,
                       label='Test Loss (at best)', color='green', marker='*',
                       s=200, zorder=5, edgecolors='darkgreen', linewidth=2)

            # Annotate best points
            for i, (ep, loss) in enumerate(zip(self.best_model_epochs, self.test_losses)):
                ax.annotate(f'Best {i + 1}\n{loss:.4f}',
                            xy=(ep, loss), xytext=(5, 5),
                            textcoords='offset points', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

        # Set axis limits with some padding
        if self.train_losses:
            all_losses = self.train_losses.copy()
            if self.val_losses:
                all_losses.extend(self.val_losses)
            if self.test_losses:
                all_losses.extend(self.test_losses)

            min_loss = min(all_losses)
            max_loss = max(all_losses)
            loss_range = max_loss - min_loss

            ax.set_ylim(min_loss - 0.1 * loss_range, max_loss + 0.1 * loss_range)
            ax.set_xlim(-1, len(self.train_losses) + 1)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Progress - All Losses', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add a secondary y-axis for better readability if losses vary significantly
        if self.train_losses and self.val_losses:
            if len(self.train_losses) > 0 and len(self.val_losses) > 0:
                train_std = np.std(self.train_losses[-min(10, len(self.train_losses)):])
                val_std = np.std(self.val_losses[-min(5, len(self.val_losses)):])

                # Add text with current values
                text_str = f'Current Epoch: {epoch}\n'
                text_str += f'Train Loss: {self.train_losses[-1]:.4f}\n'
                if self.val_losses:
                    text_str += f'Val Loss: {self.val_losses[-1]:.4f}\n'
                if self.test_losses:
                    text_str += f'Best Test: {min(self.test_losses):.4f}'

                ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
                        verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Add to tensorboard
        self.writer.add_figure('Losses/combined', fig, epoch)
        plt.close(fig)

        # Also log individual scalars for easy comparison in TensorBoard
        if self.val_losses:
            # Log on same plot using TensorBoard's scalar grouping
            self.writer.add_scalars('Losses/all', {
                'train': self.train_losses[-1],
                'validation': self.val_losses[-1]
            }, epoch)

        if self.test_losses:
            self.writer.add_scalar('Losses/test_at_best', self.test_losses[-1], epoch)

    def _evaluate_test(self, epoch, score_fn):
        """Evaluate on test set"""
        if 'test' not in self.dataloaders:
            return None

        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.dataloaders['test'], desc='Testing'):
                loss = score_fn(batch)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(self.dataloaders['test'])
        self.test_losses.append(avg_test_loss)

        # Log test loss
        self.writer.add_scalar('Loss/test_at_best', avg_test_loss, epoch)

        return avg_test_loss

    def _visualize_class_reconstructions(self, epoch, val_loss, test_loss):
        """Visualize reconstruction for each class sample"""
        if not self.class_samples:
            return

        self.model.eval()

        # Create grid of originals and reconstructions
        originals = []
        reconstructions = []

        with torch.no_grad():
            for class_idx in sorted(self.class_samples.keys())[:self.num_classes]:
                original = self.class_samples[class_idx]
                reconstructed = self.model(original)

                originals.append(original.cpu())
                reconstructions.append(reconstructed.cpu())

        if originals:
            # Stack all images
            originals = torch.cat(originals, dim=0)
            reconstructions = torch.cat(reconstructions, dim=0)

            # Create comparison grid
            comparison = torch.cat([originals, reconstructions], dim=0)
            grid = torchvision.utils.make_grid(comparison, nrow=self.num_classes, normalize=True)

            # Add text overlay with metrics
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.imshow(grid.permute(1, 2, 0).numpy())
            ax.axis('off')

            title = f'Epoch {epoch} | Val Loss: {val_loss:.4f}'
            if test_loss is not None:
                title += f' | Test Loss: {test_loss:.4f}'
            title += f' | LR: {self.optimizer.param_groups[0]["lr"]:.6f}'

            ax.set_title(title, fontsize=12, pad=20)

            # Add class labels
            fig.text(0.5, 0.05, 'Top: Original | Bottom: Reconstructed | Classes: 0-9',
                     ha='center', fontsize=10)

            # Log to tensorboard
            self.writer.add_figure(f'Reconstruction/best_model_epoch_{epoch}', fig, epoch)
            plt.close(fig)

    def _log_model_weights(self, epoch):
        """Log model weight distributions"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'Weights/{name}', param.data.cpu().numpy(), epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad.data.cpu().numpy(), epoch)

    def _log_best_model_info(self, epoch, val_loss, test_loss):
        """Log detailed info about the best model"""
        test_loss_str = f"{test_loss:.6f}" if test_loss is not None else "N/A"
        gradient_norm_str = f"{self.metrics_history['gradient_norm'][-1]:.3f}" if self.metrics_history[
            'gradient_norm'] else "N/A"
        improvement_str = f"{(self.val_losses[-2] - val_loss):.6f}" if len(self.val_losses) > 1 else "0.000000"

        info = f"""
        **Best Model Found at Epoch {epoch}**

        - Validation Loss: {val_loss:.6f}
        - Test Loss: {test_loss_str}
        - Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}
        - Training Loss: {self.train_losses[-1]:.6f}
        - Gradient Norm: {gradient_norm_str}
        - Improvement: {improvement_str}
        """

        self.writer.add_text('BestModel/Info', info, epoch)

    def _log_training_summary(self):
        """Log final training summary"""
        test_loss_str = f"{self.test_losses[-1]:.6f}" if self.test_losses else "N/A"

        summary = f"""
        **Training Summary**

        - Total Epochs: {len(self.train_losses)}
        - Best Validation Loss: {self.best_val_loss:.6f}
        - Best Model Epochs: {self.best_model_epochs}
        - Final Test Loss: {test_loss_str}
        - Final Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}
        - Average Training Time: N/A
        """

        self.writer.add_text('Training/Summary', summary, len(self.train_losses))

    def save(self, epoch, model_name: str):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_epochs': self.val_epochs,
            'test_losses': self.test_losses,
            'metrics_history': self.metrics_history,
            'best_model_epochs': self.best_model_epochs
        }, self.exp_dir / f'{model_name}.pt')

    def eval(self, epoch, score_fn: Callable):
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

            # Create side-by-side comparison
            n_samples = min(8, sample_image.size(0))
            comparison = torch.cat([sample_image[:n_samples], sample_recon[:n_samples]])
            grid = torchvision.utils.make_grid(comparison, nrow=n_samples, normalize=True)

            # Log to tensorboard
            self.writer.add_image('Samples/comparison', grid, epoch)