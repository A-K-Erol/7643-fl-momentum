import torch.optim as optim
import torch_optimizer as torch_optimizer
from config import Config

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
            return QHM(params) # Alireza
        case 'accsgd':
            return torch_optimizer.AccSGD(params, lr=0.1) # Ansel
        case _:
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
