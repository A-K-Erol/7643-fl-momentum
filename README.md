# Federated Learning with Momentum-Based Optimizers

This repository accompanies our project on analyzing convergence and efficiency in Federated Learning (FL) with heterogeneous (non-IID) data. We perform a comparison of five advanced momentum-based optimizers within the FedAvg framework:

- Polyak Momentum
- Nesterov Accelerated Gradient (NAG)
- Quasi-Hyperbolic Momentum (QHM)
- Rectified Adam (RAdam)
- Accelerated SGD (AccSGD)

Using PyTorch and the Flower FL platform, we benchmark these methods on CIFAR-10 and CIFAR-100, under both IID and Dirichlet $\alpha=0.5$ non-IID splits. Results show Polyak momentum (m=0.1) delivers the best performance in terms of convergence speed and final accuracy, with RAdam as a close second. Non-IID conditions significantly degrade performance across all optimizers.

## How to run
- Dependencies in the requirements.txt, install using `pip install -r requirements.txt`
- Edit configuration in config.py (epochs, distirbution, optimizer, etc.)
- To run a flower simulation: `python3 run.py`
    * Optional: Change logging info in run.py
- Modify and update models and optimizers in models.py and optimizers.py
- Create custom aggregation strategies in strategies.py, FedAvg is the default one.
- Official flower documentation: https://flower.ai/docs/framework/
