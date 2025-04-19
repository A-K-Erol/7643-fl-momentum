import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from optimizers import get_optimizer
import numpy as np
from collections import OrderedDict
from typing import List

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()

        k = 0 if Config.DATASET == 'cifar10' else 1
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120 + 100*k)
        self.fc2 = nn.Linear(120 + 100*k, 84 + 84*k)
        self.fc3 = nn.Linear(84 + 84*k, 10 + 90*k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(net.parameters())
    net.to(DEVICE)
    
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch["img"].to(DEVICE), batch["label" if Config.DATASET == 'cifar10' else 'fine_label'].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(DEVICE), batch["label" if Config.DATASET == 'cifar10' else 'fine_label'].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def check_centralized_accuracy(net, trainloader, valloader, testloader, epochs=5):
    """Train the network and evaluate on the test set."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net().to(DEVICE)

    for epoch in range(5):
        train(net, trainloader, 1)
        loss, accuracy = test(net, valloader)
        print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

    loss, accuracy = test(net, testloader)
    print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
    return accuracy

def get_model(name):
    if name == "net":
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
        # return model
        return Net()