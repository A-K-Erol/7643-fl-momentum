import torch
import torch.nn as nn
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
            return optim.SGD(params, lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM, nesterov=True) # Aamir
        case 'rectified_adam':
            return optim.RAdam(params) # Mike
        case 'qhm':
            return torch_optimizer.QHM(params) # Alireza
        case 'accsgd':
            return torch_optimizer.AccSGD(params, lr=0.1) # Ansel
        case _:
            print(f"Optimizer {optimizer} not recognized, using Adam by default.")

    return optim.Adam(params)