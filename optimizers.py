import torch.optim as optim

def get_adam(params):
    return optim.Adam(params)