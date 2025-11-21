from pathlib import Path
from typing import Dict, Literal, Optional
import torch
import numpy as np
from tqdm import tqdm
from collections.abc import Callable
from genesis.writers import base
from genesis.trainers.base import BaseTrainer
from genesis.factories.writers import writer_factory
from genesis.utils import helpers

class Trainer(BaseTrainer):
    """Refactored Trainer with separated logging concerns"""

    def __init__(self,
                 experiment_name: str,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 dataloaders: Dict,
                 writer: Optional[base.BaseWriter] = None,
                 scheduler: torch.optim.lr_scheduler = None,
                 num_classes: int = 10):
        """
        A Trainer for a Model whose objective is reconstruction

        Args:
            experiment_name: Name of the experiment
            model: Model to be trained
            loss_fn: Loss function
            optimizer: Optimizer to be used
            dataloaders: Dictionary with 'train', 'val', 'test' DataLoaders
            writer: BaseWriter instance for logging (if None, creates TensorBoard writer)
            scheduler: Learning rate scheduler
            num_classes: Number of classes for per-class visualization
        """
        super().__init__()

        # Core training components
        self.dataloaders: Dict[Literal['train', 'val', 'test']: torch.utils.data.DataLoader] = dataloaders
        self.model: torch.nn.Module = model
        self.loss_fn: Callable = loss_fn
        self.optimizer: torch.optim = optimizer
        self.scheduler = scheduler
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes

        # Experiment directory
        self.exp_dir = Path('experiments') / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Writer for logging (default to TensorBoard if not provided)
        if writer is None:
            self.writer = writer_factory('tensorboard', self.exp_dir, 'logs')
        else:
            self.writer = writer

        self.patience = 10
        self.patience_counter = 0

        # Training state
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.val_epochs = []
        self.best_model_epochs = []

        # Metrics history
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'learning_rate': [],
            'gradient_norm': []
        }

        # Store class samples for visualization
        self.class_samples = self._get_class_samples()

        # Log initial model info
        self._log_initial_info()

    def _get_class_samples(self) -> Dict[int, torch.Tensor]:
        """Get one sample from each class for consistent visualization"""
        class_samples = {}
        self.model.eval()

        with torch.no_grad():
            # Try validation set first
            for batch in self.dataloaders.get('test', []):
                images, labels = batch
                for i in range(len(labels)):
                    label = labels[i].item()
                    if label not in class_samples:
                        class_samples[label] = images[i:i + 1].to(self.device)
                    if len(class_samples) == self.num_classes:
                        return class_samples

            # Fall back to val set if needed
            if len(class_samples) < self.num_classes:
                for batch in self.dataloaders.get('val', []):
                    images, labels = batch
                    for i in range(len(labels)):
                        label = labels[i].item()
                        if label not in class_samples:
                            class_samples[label] = images[i:i + 1].to(self.device)
                        if len(class_samples) == self.num_classes:
                            return class_samples

        return class_samples

    def _log_initial_info(self):
        """Log initial model and experiment information"""
        # Log model architecture
        try:
            sample_batch = next(iter(self.dataloaders['train']))
            sample_input = sample_batch[0][:1].to(self.device)
            self.writer.log_model_graph(self.model, sample_input)
        except Exception as e:
            print(f"Could not log model architecture: {e}")

        # Log model parameters info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        info_text = f"""
        **Model Information**
        - Total Parameters: {total_params:,}
        - Trainable Parameters: {trainable_params:,}
        - Device: {self.device}
        - Number of Classes: {self.num_classes}
        """

        self.writer.log_text('Model/Info', info_text)

    def score_fn(self, batch):
        """Calculate loss for a batch"""
        images, _ = batch
        images = images.to(self.device)
        reconstructed,_ = self.model(images)
        loss = self.loss_fn(reconstructed, images)
        return loss

    def train(self,score_fn: Callable,epoch:int):
        self.model.train()
        self.model.to(self.device)
        epoch_train_loss = 0
        batch_losses = []
        gradient_norms = []
        progress_bar = tqdm( self.dataloaders['train'], desc = f'Training Epoch {epoch+1}')
        for idx, batch in enumerate(progress_bar):
            # Forward pass
            loss = score_fn(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Calculate gradient norm
            grad_norm = self._calculate_gradient_norm()
            gradient_norms.append(grad_norm)

            # Optimizer step
            self.optimizer.step()

            # Track losses
            batch_losses.append(loss.item())
            epoch_train_loss += loss.item()

        # Calculate epoch statistics
        avg_train_loss = epoch_train_loss / len(self.dataloaders['train'])
        avg_grad_norm = float(np.mean(gradient_norms))
        current_lr = self.optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({'Train Loss': avg_train_loss, 'Grad Norm': avg_grad_norm, 'LR': current_lr})

        return avg_train_loss, avg_grad_norm, current_lr

    def eval(self, epoch: int, score_fn: Callable) -> float:
        """Evaluate model on validation set"""
        self.model.eval()
        epoch_val_loss = 0

        with torch.no_grad():
            progress_bar = tqdm(self.dataloaders['val'], desc=f'Validating Epoch {epoch + 1}')
            for batch in progress_bar:
                loss = score_fn(batch)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(self.dataloaders['val'])
        progress_bar.set_postfix({'Val Loss': avg_val_loss})
        self.val_losses.append(avg_val_loss)

        return avg_val_loss

    def run(self,
            epochs: int,
            train_fn: Callable = None,
            score_fn: Callable = None,
            save_fn: Callable = None,
            eval_fn: Callable = None,
            visualize: bool = True,
            visualization_step: int = 10):
        """
        Main training loop

        Args:
            epochs: Number of epochs to train
            train_fn: Function to perform training (defaults to self.train)
            score_fn: Function to calculate loss (defaults to self.score_fn)
            save_fn: Function to save model (defaults to self.save)
            eval_fn: Function to evaluate model (defaults to self.eval)
            visualize: Whether to perform visualization
            visualization_step: How often to visualize
        """
        # Set default functions
        score_fn = self.score_fn if score_fn is None else score_fn
        save_fn = self.save if save_fn is None else save_fn
        train_fn = self.train if train_fn is None else train_fn
        eval_fn = self.eval if eval_fn is None else eval_fn

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Visualization every {visualization_step} epochs")
        print("=" * 50)

        for epoch in range(epochs):
            # Training phase

            avg_train_loss, avg_grad_norm, current_lr = train_fn(score_fn=score_fn,epoch=epoch)

            # Update history
            self.train_losses.append(avg_train_loss)
            self.metrics_history['train_loss'].append(avg_train_loss)
            self.metrics_history['gradient_norm'].append(avg_grad_norm)
            self.metrics_history['learning_rate'].append(current_lr)

            # Log epoch metrics
            self.writer.log_scalar('Loss/train_epoch', avg_train_loss, epoch)
            self._log_training_progress(epoch)

            # Validation phase

            avg_val_loss = eval_fn(epoch, score_fn)
            self.val_epochs.append(epoch)
            self.metrics_history['val_loss'].append(avg_val_loss)

            self.writer.log_scalar('Loss/val_epoch', avg_val_loss, epoch)

            # # Update progress bar

            # Learning rate scheduler step
            if self.scheduler is not None:
                self.scheduler.step(avg_val_loss)

            # Check for best model
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model_epochs.append(epoch)

                # Test on best model
                test_loss = self._evaluate_test(epoch, score_fn)
                if test_loss is not None:
                    self.metrics_history['test_loss'].append(test_loss)

                # Visualize best model reconstructions
                self._visualize_best_model(epoch, avg_val_loss, test_loss)

                # Log model weights
                # self.writer.log_model_weights(self.model, epoch)

                # Save best model
                save_fn(epoch, "best_model.pt")
                print(f"✓ Saved best model with validation loss: {self.best_val_loss:.4f} ")

                # Log best model info
                self._log_best_model_info(epoch, avg_val_loss, test_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            # Checkpoint saving
            if (epoch + 1) % 50 == 0:
                save_fn(epoch, f"checkpoint_epoch_{epoch}.pt")

            # Print epoch summary
            self._print_epoch_summary(epoch, epochs, avg_train_loss,
                                      avg_val_loss,
                                      test_loss,
                                      avg_grad_norm, current_lr)

        # Training complete
        save_fn(epochs - 1, "final_model.pt")
        self._log_final_summary()
        self.writer.close()

        print(f"\n✓ Training complete! Results saved to {self.exp_dir}")
        print(f"✓ Best validation loss: {self.best_val_loss:.4f}")

        return self.exp_dir

    def _calculate_gradient_norm(self) -> float:
        """Calculate the gradient norm across all parameters"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5





    def _evaluate_test(self, epoch: int, score_fn: Callable) -> Optional[float]:
        """Evaluate on test set"""
        if 'test' not in self.dataloaders:
            return None

        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            progress_bar = tqdm(self.dataloaders['test'], desc=f'Testing Epoch {epoch + 1}')
            for batch in progress_bar:
                loss = score_fn(batch)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(self.dataloaders['test'])
        self.test_losses.append(avg_test_loss)
        self.writer.log_scalar('Loss/test_at_best', avg_test_loss, epoch)

        return avg_test_loss

    def _log_training_progress(self, epoch: int):
        """Log training progress with combined loss curves"""

        # Also log as scalars for easy comparison
        if self.val_losses:
            self.writer.log_scalars('Losses/all', {
                'train': self.train_losses[-1],
                'validation': self.val_losses[-1]
            }, epoch)

    def _visualize_best_model(self, epoch: int, val_loss: float, test_loss: Optional[float]):
        """Visualize reconstructions for the best model"""
        if not self.class_samples:
            return

        self.model.eval()

        with torch.no_grad():
            originals = []
            reconstructions = []
            bottlenecks = []

            for class_idx in sorted(self.class_samples.keys())[:self.num_classes]:
                original = self.class_samples[class_idx]
                reconstructed,bottleneck = self.model(original)
                originals.append(original)
                reconstructions.append(reconstructed)
                bottlenecks.append(bottleneck)

            if originals:
                originals = torch.cat(originals, dim=0)
                reconstructions = torch.cat(reconstructions, dim=0)
                bottlenecks = torch.cat(bottlenecks, dim=0)

                title = f"Epoch {epoch} | Val: {val_loss:.4f}"
                if test_loss is not None:
                    title += f" | Test: {test_loss:.4f}"

                self.writer.log_reconstruction_comparison(
                    originals, reconstructions, epoch,
                    num_samples=len(originals), title=title
                )

                helpers.visualize_bottleneck(self.model,self.dataloaders['test'], self.writer,epoch)

    def _log_best_model_info(self, epoch: int, val_loss: float, test_loss: Optional[float]):
        """Log information about the best model"""
        test_loss = f"{test_loss:.2f}" if test_loss is not None else 'N/A'
        info = f"""
        **Best Model Found at Epoch {epoch}**
        - Validation Loss: {val_loss:.2f}
        - Test Loss: {test_loss}
        - Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}
        - Training Loss: {self.train_losses[-1]:.2f}
        """
        self.writer.log_text('BestModel/Info', info, epoch)

    def _log_final_summary(self):
        """Log final training summary"""
        test_losses = f"{self.test_losses[-1]:.2f}" if self.test_losses else 'N/A'
        summary = f"""
        **Training Summary**
        - Total Epochs: {len(self.train_losses)}
        - Best Validation Loss: {self.best_val_loss:.2f}
        - Best Model Epochs: {self.best_model_epochs}
        - Final Test Loss: {test_losses} 
        """
        self.writer.log_text('Training/Summary', summary)

    def _print_epoch_summary(self, epoch: int, total_epochs: int, train_loss: float,
                             val_loss: Optional[float], test_loss:Optional[float],grad_norm: float, lr: float):
        """Print epoch summary to console"""
        print(f"\nEpoch {epoch + 1}/{total_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        if val_loss is not None:
            print(f"  Val Loss: {val_loss:.4f}")
        if test_loss is not None:
            print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Gradient Norm: {grad_norm:.3f}")
        print(f"  Learning Rate: {lr:.6f}")
        print("-" * 50)

    def save(self, epoch: int, model_name: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'test_losses': self.test_losses,
            'metrics_history': self.metrics_history
        }

        save_path = self.exp_dir / f'{model_name}'
        torch.save(checkpoint, save_path)