import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from pathlib import Path
from PIL import Image


class AnimalDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = Path(root_path)
        self.transform = transform or transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
             ]
        )

        self.images = []
        self.labels = []
        self.class_names = []

        for folder in self.root_path.iterdir():
            if folder.is_dir():
                class_idx = len(self.class_names)
                self.class_names.append(folder.name)

                for img_path in folder.glob("*.jp*g"):
                    self.images.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def show_tensor(tensor):
    """Convert tensor to PIL image and display"""
    to_pil = ToPILImage()
    img = to_pil(tensor)
    return img
