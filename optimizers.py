import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as torch_optimizer
from config import Config
from torch.optim import Optimizer

def get_optimizer(params, optimizer=Config.OPTIMIZER):
    if optimizer == 'sgd':
        return optim.SGD(params, lr=0.01)
    elif optimizer == 'polyak':
        return optim.SGD(params, lr=0.01, momentum=0.9)
    elif optimizer == 'nesterov':
        return optim.SGD(params, lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM, nesterov=True)  # Aamir
    elif optimizer == 'rectified_adam':
        return optim.RAdam(params,
                               lr= 0.0003,
                               betas=(0.90, 0.999),
                               weight_decay=0.0004,
                               eps=1e-08
                               ) # Mike
    elif optimizer == 'qhm':
        return QHM(params, momentum=Config.MOMENTUM, nu=1.0)
    elif optimizer == 'accsgd':
        return torch_optimizer.AccSGD(params, lr=0.1, weight_decay =0.0001)
    else:
        print(f"Optimizer {optimizer} not recognized, using Adam by default.")
        return optim.Adam(params)



class QHM(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, nu=1.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= nu <= 1.0:
            raise ValueError(f"Invalid nu value: {nu}")

        defaults = dict(lr=lr, momentum=momentum, nu=nu)
        super(QHM, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if 'momentum_buffer' not in self.state[p]:
                    buf = self.state[p]['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = self.state[p]['momentum_buffer']
                    buf.mul_(group['momentum']).add_(d_p)

                p.data.add_(-group['lr'], (1 - group['nu']) * d_p + group['nu'] * buf)

        return loss
