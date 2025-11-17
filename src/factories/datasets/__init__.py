from torchvision.transforms import ToPILImage
import torch

def plot_image(tensor:torch.Tensor):
    """Convert tensor to PIL image and display"""
    to_pil = ToPILImage()
    img = to_pil(tensor)
    return img