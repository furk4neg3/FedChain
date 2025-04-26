import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Net

# Load final global model
model = Net()
model.load_state_dict(torch.load("final_global_model.pth"))

# Load test dataset
transform = transforms.Compose([transforms.ToTensor()])
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transform),
    batch_size=64, shuffle=False
)

# Evaluate model accuracy
model.eval()
correct = total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = correct / total * 100
print(f"Final Global Model Accuracy: {accuracy:.2f}%")
