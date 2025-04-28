import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as torch_optimizer
from config import Config
from torch.optim.optimizer import Optimizer

def get_optimizer(params, optimizer=Config.OPTIMIZER):
    match optimizer:
        case 'sgd':
            return optim.SGD(params, lr=0.01)
        case 'polyak':
            return optim.SGD(params, lr=0.01, momentum=0.9)
        case 'nesterov':
            return optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True) # Aamir
        case 'rectified_adam':
            return optim.RAdam(params) # Mike
        case 'qhm':
            return torch_optimizer.QHM(params) # Alireza
        case 'accsgd':
            return torch_optimizer.AccSGD(params, lr=0.1) # Ansel
        case _:
            print(f"Optimizer {optimizer} not recognized, using Adam by default.")


    return optim.Adam(params)

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
