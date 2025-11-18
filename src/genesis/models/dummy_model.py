import torch

class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = torch.nn.Linear(28,28)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.linear(x)
        output = self.relu(output)
        return output