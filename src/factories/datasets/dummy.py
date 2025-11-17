import torch

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(DummyDataset, self).__init__()
        self.x = torch.randn(1000, 28)
        self.y = torch.randn(1000)

    def __len__(self):
        return 1000
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_data_loader():
    return {
        'train':torch.utils.data.DataLoader(
        dataset=DummyDataset(),
        batch_size=100,
        shuffle=True),
        'val':torch.utils.data.DataLoader(
        dataset=DummyDataset(),
            batch_size=100,
            shuffle=True),
        'test':torch.utils.data.DataLoader(
            dataset=DummyDataset(),
            batch_size=100,
        shuffle = True),
    }