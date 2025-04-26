# client/model.py

import torch
import torch.nn as nn

class Net(nn.Module):
    """Same model definition for clients."""
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)

def client_update(model, optimizer, train_loader, epochs=1):
    """Train `model` on `train_loader` for `epochs` epochs."""
    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(data), target)
            loss.backward()
            optimizer.step()
