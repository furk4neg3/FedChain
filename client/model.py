# client.py
import flwr as fl
from model import Net, client_update, train_loader

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
        # Optional evaluation method
        pass

fl.client.start_numpy_client(server_address="192.168.X.X:8080", client=MNISTClient())
