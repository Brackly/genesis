from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

class BaseWriter(ABC):
    """Abstract base class for experiment logging"""

    @abstractmethod
    def __init__(self, log_dir: Union[str, Path], experiment_name: str = None):
        """Initialize the writer with a logging directory"""
        pass

    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value"""
        pass

    @abstractmethod
    def log_scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        """Log multiple scalars on the same plot"""
        pass

    @abstractmethod
    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log a histogram of values"""
        pass

    @abstractmethod
    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: int) -> None:
        """Log a single image"""
        pass

    @abstractmethod
    def log_images(self, tag: str, images: Union[torch.Tensor, np.ndarray], step: int, **kwargs) -> None:
        """Log multiple images as a grid"""
        pass

    @abstractmethod
    def log_figure(self, tag: str, figure: plt.Figure, step: int) -> None:
        """Log a matplotlib figure"""
        pass

    @abstractmethod
    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        """Log text"""
        pass

    @abstractmethod
    def log_model_graph(self, model: torch.nn.Module, input_sample: torch.Tensor) -> None:
        """Log model architecture graph"""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the writer and clean up resources"""
        pass

    # High-level convenience methods (can be overridden for custom behavior)
    def log_reconstruction_comparison(self,
                                      originals: torch.Tensor,
                                      reconstructions: torch.Tensor,
                                      step: int,
                                      num_samples: int = 8,
                                      title: str = None) -> None:
        """Log side-by-side comparison of originals and reconstructions"""
        pass

    def log_loss_curves(self,
                        losses_dict: Dict[str, List[float]],
                        step: int,
                        title: str = "Training Progress") -> None:
        """Log multiple loss curves on the same plot"""
        pass

    def log_model_weights(self, model: torch.nn.Module, step: int) -> None:
        """Log model weight distributions"""
        pass


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