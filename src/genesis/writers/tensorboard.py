from pathlib import Path
from typing import Dict, List,  Union
import numpy as np
import torch
import matplotlib.pyplot as plt

from genesis.writers.base import BaseWriter

class TensorBoardWriter(BaseWriter):
    """TensorBoard implementation of BaseWriter"""

    def __init__(self, log_dir: Union[str, Path], experiment_name: str = None):
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = Path(log_dir)
        if experiment_name:
            self.log_dir = self.log_dir / experiment_name

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        self.writer.add_scalars(tag, values, step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: int) -> None:
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        self.writer.add_image(tag, image, step)

    def log_images(self, tag: str, images: Union[torch.Tensor, np.ndarray], step: int, **kwargs) -> None:
        import torchvision
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        # Create grid
        nrow = kwargs.get('nrow', 8)
        normalize = kwargs.get('normalize', True)
        grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=normalize)
        self.writer.add_image(tag, grid, step)

    def log_figure(self, tag: str, figure: plt.Figure, step: int) -> None:
        self.writer.add_figure(tag, figure, step)
        plt.close(figure)  # Clean up to prevent memory leaks

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        self.writer.add_text(tag, text, step)

    def log_model_graph(self, model: torch.nn.Module, input_sample: torch.Tensor) -> None:
        try:
            self.writer.add_graph(model, input_sample)
        except Exception as e:
            print(f"Could not log model graph: {e}")

    def log_reconstruction_comparison(self,
                                      originals: torch.Tensor,
                                      reconstructions: torch.Tensor,
                                      step: int,
                                      num_samples: int = 8,
                                      title: str = None) -> None:
        import torchvision

        # Limit to num_samples
        originals = originals[:num_samples]
        reconstructions = reconstructions[:num_samples]

        # Create comparison grid
        comparison = torch.cat([originals, reconstructions], dim=0)
        grid = torchvision.utils.make_grid(comparison, nrow=num_samples, normalize=True)

        # Add to tensorboard
        tag = f"Reconstruction/{title}" if title else "Reconstruction/comparison"
        self.writer.add_image(tag, grid, step)

    def log_loss_curves(self,
                        losses_dict: Dict[str, List[float]],
                        step: int,
                        title: str = "Training Progress") -> None:
        """Create and log a multi-line loss plot"""
        fig, ax = plt.subplots(figsize=(10, 6))

        for name, values in losses_dict.items():
            if values:  # Only plot if there are values
                ax.plot(range(len(values)), values, label=name, linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.log_figure('Losses/combined', fig, step)

    def log_model_weights(self, model: torch.nn.Module, step: int) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'Weights/{name}', param.data.cpu().numpy(), step)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad.data.cpu().numpy(), step)

    def close(self) -> None:
        self.writer.close()




class NoOpWriter(BaseWriter):
    """No-operation writer for testing or when logging is disabled"""

    def __init__(self, log_dir: Union[str, Path] = None, experiment_name: str = None):
        pass

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        pass

    def log_scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        pass

    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        pass

    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: int) -> None:
        pass

    def log_images(self, tag: str, images: Union[torch.Tensor, np.ndarray], step: int, **kwargs) -> None:
        pass

    def log_figure(self, tag: str, figure: plt.Figure, step: int) -> None:
        plt.close(figure)  # Still close to prevent memory leaks

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        pass

    def log_model_graph(self, model: torch.nn.Module, input_sample: torch.Tensor) -> None:
        pass

    def close(self) -> None:
        pass


class CompositeWriter(BaseWriter):
    """Composite writer that logs to multiple backends simultaneously"""

    def __init__(self, writers: List[BaseWriter]):
        self.writers = writers

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        for writer in self.writers:
            writer.log_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        for writer in self.writers:
            writer.log_scalars(tag, values, step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        for writer in self.writers:
            writer.log_histogram(tag, values, step)

    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: int) -> None:
        for writer in self.writers:
            writer.log_image(tag, image, step)

    def log_images(self, tag: str, images: Union[torch.Tensor, np.ndarray], step: int, **kwargs) -> None:
        for writer in self.writers:
            writer.log_images(tag, images, step, **kwargs)

    def log_figure(self, tag: str, figure: plt.Figure, step: int) -> None:
        for writer in self.writers:
            writer.log_figure(tag, figure, step)

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        for writer in self.writers:
            writer.log_text(tag, text, step)

    def log_model_graph(self, model: torch.nn.Module, input_sample: torch.Tensor) -> None:
        for writer in self.writers:
            writer.log_model_graph(model, input_sample)

    def log_reconstruction_comparison(self, originals: torch.Tensor, reconstructions: torch.Tensor,
                                      step: int, num_samples: int = 8, title: str = None) -> None:
        for writer in self.writers:
            if hasattr(writer, 'log_reconstruction_comparison'):
                writer.log_reconstruction_comparison(originals, reconstructions, step, num_samples, title)

    def log_loss_curves(self, losses_dict: Dict[str, List[float]], step: int, title: str = "Training Progress") -> None:
        for writer in self.writers:
            if hasattr(writer, 'log_loss_curves'):
                writer.log_loss_curves(losses_dict, step, title)

    def log_model_weights(self, model: torch.nn.Module, step: int) -> None:
        for writer in self.writers:
            if hasattr(writer, 'log_model_weights'):
                writer.log_model_weights(model, step)

    def close(self) -> None:
        for writer in self.writers:
            writer.close()


# Factory function for easy writer creation
def create_writer(writer_type: str,
                  log_dir: Union[str, Path],
                  experiment_name: str = None,
                  **kwargs) -> BaseWriter:
    """
    Factory function to create writers

    Args:
        writer_type: One of 'tensorboard', 'wandb', 'noop', or 'composite'
        log_dir: Directory for logging
        experiment_name: Name of the experiment
        **kwargs: Additional arguments for specific writers

    Returns:
        Instance of BaseWriter
    """
    writers = {
        'tensorboard': TensorBoardWriter,
        'wandb': WandBWriter,
        'noop': NoOpWriter
    }

    if writer_type == 'composite':
        # Create multiple writers
        writer_list = []
        for wtype in kwargs.get('writers', ['tensorboard']):
            writer_list.append(writers[wtype](log_dir, experiment_name))
        return CompositeWriter(writer_list)

    if writer_type not in writers:
        raise ValueError(f"Unknown writer type: {writer_type}. Choose from {list(writers.keys())}")

    return writers[writer_type](log_dir, experiment_name, **kwargs)


# Example usage
if __name__ == "__main__":
    # Create a TensorBoard writer
    writer = create_writer('tensorboard', 'experiments', 'my_experiment')

    # Or create a W&B writer
    # writer = create_writer('wandb', 'experiments', 'my_experiment', project='my_project')

    # Or log to both simultaneously
    # writer = create_writer('composite', 'experiments', 'my_experiment', writers=['tensorboard', 'wandb'])

    # Use in training loop
    for epoch in range(10):
        # Log scalars
        writer.log_scalar('loss/train', np.random.random(), epoch)
        writer.log_scalar('loss/val', np.random.random(), epoch)

        # Log multiple scalars
        writer.log_scalars('metrics', {
            'accuracy': np.random.random(),
            'precision': np.random.random()
        }, epoch)

        # Log histograms
        writer.log_histogram('weights', np.random.randn(100), epoch)

    writer.close()