
from typing import Dict, List, Optional, Any, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from base import BaseWriter


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