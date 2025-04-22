import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DFedAvgM_Optimizer:
    def __init__(self, model, lr=0.01, momentum=0.9, num_local_updates=5):
        """
        Implements Decentralized Federated Averaging with Momentum (DFedAvgM).
        
        Args:
        - model (torch.nn.Module): The local model for each client.
        - lr (float): Learning rate for SGD.
        - momentum (float): Momentum parameter.
        - num_local_updates (int): Number of local SGD iterations before communication.
        """
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.num_local_updates = num_local_updates

    def local_update(self, train_loader, loss_fn, device="cpu"):
        """
        Performs local training updates before communication.

        Args:
        - train_loader (torch.utils.data.DataLoader): Local dataset loader.
        - loss_fn (torch.nn.Module): Loss function.
        - device (str): Device to run the computation (e.g., "cpu" or "cuda").
        """
        self.model.train()
        self.model.to(device)

        for _ in range(self.num_local_updates):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                # Forward pass
                output = self.model(data)
                loss = loss_fn(output, target)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
    def get_model_params(self):
        """Returns model parameters as a flat tensor."""
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])

    def set_model_params(self, param_tensor):
        """Sets model parameters from a flat tensor."""
        index = 0
        for param in self.model.parameters():
            param_length = param.numel()
            param.data.copy_(param_tensor[index:index + param_length].view(param.shape))
            index += param_length
import torch
from torch.optim.optimizer import Optimizer

class QHM(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, nu=1.0):
        assert 0.0 <= nu <= 1.0, "nu must be in [0,1]"
        assert 0.0 <= momentum < 1.0, "momentum must be in [0,1)"
        defaults = dict(lr=lr, momentum=momentum, nu=nu)
        super(QHM, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']
            beta = group['momentum']
            nu = group['nu']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                buf = state['momentum_buffer']

                buf.mul_(beta).add_(grad, alpha=1 - beta)
                update = nu * grad + (1 - nu) * buf
                p.data.add_(-lr, update)
        return loss
