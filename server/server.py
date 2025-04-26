#!/usr/bin/env python3
import os

import pickle
import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays
from blockchain_interface import submit_model_hash
from model import Net
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Number of federated clients (must match your docker-compose)
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "3"))

# Prepare full MNIST test set for global evaluation
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def evaluate_global(model: Net) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, pred = torch.max(output, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
    return 100.0 * correct / total

class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        # 1) Default aggregation returns (Parameters, metrics)
        parameters, metrics = super().aggregate_fit(rnd, results, failures)

        # 2) Convert Parameters → numpy arrays for hashing
        ndarrays = parameters_to_ndarrays(parameters)
        serialized = pickle.dumps([arr.tolist() for arr in ndarrays])

        # 3) Submit the hash to blockchain
        submit_model_hash(serialized)

        # 4) Evaluate global model on full test set
        global_model = Net()
        # Turn our ndarrays back into a state_dict
        state_dict = {
            k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), ndarrays)
        }
        global_model.load_state_dict(state_dict, strict=True)
        acc = evaluate_global(global_model)
        print(f"[Server] Global Accuracy after round {rnd}: {acc:.2f}%")

        # 5) Save the final global weights
        torch.save(global_model.state_dict(), "final_global_model.pth")

        return parameters, metrics

if __name__ == "__main__":
    # Require every client each round
    strategy = CustomFedAvg(
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
