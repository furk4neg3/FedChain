import flwr as fl
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from model import Net, client_update
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()
SERVER_IP = os.getenv("SERVER_IP")
SERVER_PORT = "8080"
CLIENT_ID = os.getenv("CLIENT_ID", "0")  # For logging

# Dataset setup
transform = transforms.Compose([transforms.ToTensor()])
full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

# Simulate non-IID data: split dataset manually per client ID
num_clients = int(os.getenv("NUM_CLIENTS", "3"))
data_per_client = len(full_dataset) // num_clients
start_idx = int(CLIENT_ID) * data_per_client
end_idx = start_idx + data_per_client
client_data = Subset(full_dataset, range(start_idx, end_idx))
train_loader = DataLoader(client_data, batch_size=32, shuffle=True)

# Test set for evaluation
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=32)

def test_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    accuracy = 100.0 * correct / total
    return accuracy

# Flower client
class MNISTClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = Net()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.parameters()]

    def fit(self, parameters, config):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        client_update(self.model, optimizer, train_loader, epochs=2)
        return self.get_parameters(config), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param)
        acc = test_model(self.model)
        print(f"[CLIENT {CLIENT_ID}] Accuracy: {acc:.2f}%")
        return 0.0, len(test_loader.dataset), {"accuracy": acc}

# Run client
fl.client.start_numpy_client(server_address=f"{SERVER_IP}:{SERVER_PORT}", client=MNISTClient())
