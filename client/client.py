import flwr as fl
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from model import Net, client_update
import os
from dotenv import load_dotenv

load_dotenv()
SERVER_IP = os.getenv("SERVER_IP", "server")
SERVER_PORT = os.getenv("SERVER_PORT", "8080")
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "3"))

# Partition MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
size = len(dataset) // NUM_CLIENTS
start, end = CLIENT_ID * size, (CLIENT_ID + 1) * size
subset = Subset(dataset, range(start, end))
train_loader = DataLoader(subset, batch_size=32, shuffle=True)

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            out = model(data)
            _, pred = torch.max(out, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
    return 100 * correct / total

class MNISTClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = Net()

    def get_parameters(self, config):
        # Detach to avoid "requires grad" error
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def fit(self, parameters, config):
        # Load global parameters
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param)
        # Local training
        opt = torch.optim.SGD(self.model.parameters(), lr=0.01)
        client_update(self.model, opt, train_loader, epochs=2)
        # Print local accuracy
        acc = evaluate(self.model, train_loader)
        print(f"[Client {CLIENT_ID}] Local Accuracy: {acc:.2f}%")
        # Return updated parameters
        return self.get_parameters(config), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Not used
        return 0.0, len(train_loader.dataset), {}

if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address=f"{SERVER_IP}:{SERVER_PORT}",
        client=MNISTClient()
    )
