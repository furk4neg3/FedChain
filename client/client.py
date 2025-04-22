import flwr as fl
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Net, client_update

SERVER_IP = "SERVER_IP_HERE"  # <-- Replace this
SERVER_PORT = "8080"

transform = transforms.Compose([transforms.ToTensor()])
local_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(local_data, batch_size=32, shuffle=True)

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
        return 0.0, len(train_loader.dataset), {}

fl.client.start_numpy_client(server_address=f"{SERVER_IP}:{SERVER_PORT}", client=MNISTClient())
