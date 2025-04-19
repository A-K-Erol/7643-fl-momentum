from optimizers import get_optimizer
from models import Net


net = Net()
for opt in ['sgd', 'polyak', 'nesterov', 'rectified_adam', 'qhm', 'accsgd']:
    print(f"Optimizer: {opt}")
    optimizer = get_optimizer(net.parameters(), opt)
    print(optimizer)  # This will print the optimizer object