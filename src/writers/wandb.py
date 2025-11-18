from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.writers.base import BaseWriter

class WandBWriter(BaseWriter):
    """Weights & Biases implementation of BaseWriter"""

    def __init__(self, log_dir: Union[str, Path], experiment_name: str = None, project: str = None,
                 config: Dict = None):
        import wandb

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize W&B
        self.run = wandb.init(
            project=project or "ml-experiments",
            name=experiment_name,
            config=config or {},
            dir=str(self.log_dir)
        )

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        import wandb
        wandb.log({tag: value}, step=step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        import wandb
        # Flatten the tag structure for W&B
        log_dict = {f"{tag}/{key}": val for key, val in values.items()}
        wandb.log(log_dict, step=step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        import wandb
        wandb.log({tag: wandb.Histogram(values)}, step=step)

    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: int) -> None:
        import wandb

        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # Convert to HWC format if needed
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))

        wandb.log({tag: wandb.Image(image)}, step=step)

    def log_images(self, tag: str, images: Union[torch.Tensor, np.ndarray], step: int, **kwargs) -> None:
        import wandb

        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()

        # Convert batch of images to list of images
        image_list = []
        for i in range(images.shape[0]):
            img = images[i]
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            image_list.append(wandb.Image(img))

        wandb.log({tag: image_list}, step=step)

    def log_figure(self, tag: str, figure: plt.Figure, step: int) -> None:
        import wandb
        wandb.log({tag: wandb.Image(figure)}, step=step)
        plt.close(figure)

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        import wandb
        # W&B doesn't have direct text logging like TensorBoard
        # We can use a Table or just log as a string
        wandb.log({tag: text}, step=step)

    def log_model_graph(self, model: torch.nn.Module, input_sample: torch.Tensor) -> None:
        import wandb
        # W&B watches model gradients automatically
        wandb.watch(model, log='all')

    def log_reconstruction_comparison(self,
                                      originals: torch.Tensor,
                                      reconstructions: torch.Tensor,
                                      step: int,
                                      num_samples: int = 8,
                                      title: str = None) -> None:
        import wandb

        # Create side-by-side images for W&B
        images = []
        for i in range(min(num_samples, originals.shape[0])):
            orig = originals[i].cpu().numpy()
            recon = reconstructions[i].cpu().numpy()

            # Transpose if needed
            if orig.ndim == 3 and orig.shape[0] in [1, 3]:
                orig = np.transpose(orig, (1, 2, 0))
                recon = np.transpose(recon, (1, 2, 0))

            # Concatenate side by side
            comparison = np.concatenate([orig, recon], axis=1)
            images.append(wandb.Image(comparison, caption=f"Sample {i}: Original | Reconstructed"))

        tag = f"{title}" if title else "reconstruction_comparison"
        wandb.log({tag: images}, step=step)

    def log_loss_curves(self,
                        losses_dict: Dict[str, List[float]],
                        step: int,
                        title: str = "Training Progress") -> None:
        """Log losses - W&B handles plotting automatically"""
        import wandb

        # Log the current values
        current_values = {}
        for name, values in losses_dict.items():
            if values:
                current_values[f"loss/{name}"] = values[-1]

        if current_values:
            wandb.log(current_values, step=step)

    def log_model_weights(self, model: torch.nn.Module, step: int) -> None:
        import wandb
        # W&B handles this through wandb.watch()
        # But we can also log histograms manually
        for name, param in model.named_parameters():
            if param.requires_grad:
                wandb.log({f"weights/{name}": wandb.Histogram(param.data.cpu().numpy())}, step=step)
                if param.grad is not None:
                    wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.data.cpu().numpy())}, step=step)

    def close(self) -> None:
        import wandb
        wandb.finish()