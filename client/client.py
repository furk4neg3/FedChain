# client/client.py

import os
from dotenv import load_dotenv

import flwr as fl
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from model import Net, client_update

# ─── Load environment variables ─────────────────────────────────────────────────
load_dotenv()
SERVER_IP   = os.getenv("SERVER_IP", "server")
SERVER_PORT = os.getenv("SERVER_PORT", "8080")
CLIENT_ID   = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "3"))

# ─── Prepare data loaders ───────────────────────────────────────────────────────
transform = transforms.Compose([transforms.ToTensor()])

# 1) Each client gets a non-overlapping slice of the **training** set
full_train   = datasets.MNIST("./data", train=True,  download=True, transform=transform)
slice_size   = len(full_train) // NUM_CLIENTS
start, end   = CLIENT_ID * slice_size, (CLIENT_ID + 1) * slice_size
train_subset = Subset(full_train, range(start, end))
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

# 2) All clients share the same **global test** set for evaluation
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ─── Evaluation helpers ─────────────────────────────────────────────────────────
def evaluate_accuracy(model: Net, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            total += target.size(0)
            correct += (preds == target).sum().item()
    return 100.0 * correct / total

def evaluate_loss(model: Net, loader: DataLoader) -> float:
    model.eval()
    loss = 0.0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            outputs = model(data)
            loss += F.cross_entropy(outputs, target, reduction="sum").item()
            total += target.size(0)
    return loss / total

# ─── Flower client ──────────────────────────────────────────────────────────────
class MNISTClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = Net()

    def get_parameters(self, config):
        # Detach tensors to NumPy arrays
        return [param.detach().cpu().numpy() for param in self.model.parameters()]

    def fit(self, parameters, config):
        # 1) Load server parameters
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param)

        # 2) Local training on this client's slice
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        client_update(self.model, optimizer, train_loader, epochs=2)

        # 3) Evaluate on the global test set (to show local generalization)
        local_test_acc = evaluate_accuracy(self.model, test_loader)
        print(f"[Client {CLIENT_ID}] Test-set Accuracy: {local_test_acc:.2f}%")

        # 4) Return updated parameters, number of samples, and empty metrics
        return self.get_parameters(config), len(train_subset), {}

    def evaluate(self, parameters, config):
        # Load parameters into model
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param)

        # Compute loss & accuracy on global test set
        loss = evaluate_loss(self.model, test_loader)
        acc  = evaluate_accuracy(self.model, test_loader)
        print(f"[Client {CLIENT_ID}] Remote evaluate loss={loss:.4f}, acc={acc:.2f}%")

        # **Return in order (loss, num_examples, metrics)**
        return loss, len(test_dataset), {"accuracy": float(acc)}

# ─── Start client ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address=f"{SERVER_IP}:{SERVER_PORT}",
        client=MNISTClient(),
    )
