# client/client.py

import os
import flwr as fl
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from model import Net


# ── Training / Testing ─────────────────────────────────────────────────────────
def train(model, loader, device):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()


def test(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


# ── Flower Client ──────────────────────────────────────────────────────────────
class MNISTClient(fl.client.NumPyClient):
    def __init__(self, cid, train_loader, test_loader, device):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model = Net().to(self.device)
        # ── Track rounds locally ──────────────────
        self.fit_round = 0
        self.eval_round = 0

    def get_parameters(self, config):
        # Detach before converting to numpy
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def fit(self, parameters, config):
        # Load global parameters
        for p, newp in zip(self.model.parameters(), parameters):
            p.data = torch.from_numpy(newp).to(self.device)
        # Train
        train(self.model, self.train_loader, self.device)
        # Evaluate on own test set
        loss, acc = test(self.model, self.test_loader, self.device)
        # ── Print with correct round ──────────────────
        self.fit_round += 1
        print(f"[Client {self.cid}] Local Model After Round {self.fit_round} → test-accuracy: {acc * 100:.2f}%")
        # Return updated parameters
        return self.get_parameters(config), len(self.train_loader.dataset), {"accuracy": acc}

    def evaluate(self, parameters, config):
        # Load global parameters
        for p, newp in zip(self.model.parameters(), parameters):
            p.data = torch.from_numpy(newp).to(self.device)
        # Evaluate on full test set
        loss, acc = test(self.model, self.test_loader, self.device)
        # ── Print with correct round ──────────────────
        self.eval_round += 1
        #print(f"[Client {self.cid}] Eval Round {self.eval_round} → test-accuracy: {acc * 100:.2f}%")
        return float(loss), len(self.test_loader.dataset), {"accuracy": acc}


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_data(cid: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST("data", train=True, download=True, transform=transform)
    full_test = datasets.MNIST("data", train=False, download=True, transform=transform)

    # Partition TRAIN by label; TEST is full
    if cid == 0:
        classes = [0, 1, 2]
    elif cid == 1:
        classes = [3, 4, 5, 6]
    else:
        classes = [7, 8, 9]

    train_idx = [i for i, (_, y) in enumerate(full_train) if y in classes]
    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=32, shuffle=True)
    test_loader = DataLoader(full_test, batch_size=1000, shuffle=False)

    return train_loader, test_loader


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Which shard am I?
    cid = int(os.environ.get("CLIENT_ID", "0"))

    # Compose passes SERVER_IP and SERVER_PORT
    server_ip = os.environ.get("SERVER_IP", "server")
    server_port = os.environ.get("SERVER_PORT", "8080")
    server_address = f"{server_ip}:{server_port}"

    # Load data shard
    train_loader, test_loader = load_data(cid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start the Flower client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=MNISTClient(cid, train_loader, test_loader, device),
    )
