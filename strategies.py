from flwr.server.strategy import FedAvg, FedAdam, Strategy
import flwr as fl
from flwr.common import Parameters, FitRes, EvaluateIns
import tensorflow as tf

def get_fedavg_strat(eval_strat):
    strategy = FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=10,  # Never sample less than 10 clients for training
        min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
        min_available_clients=10,  # Wait until all 10 clients are available
        evaluate_metrics_aggregation_fn=eval_strat,  # Use the weighted average as evaluation strategy
        fit_metrics_aggregation_fn=eval_strat,  # Use the weighted average as training strategy
    )
    return strategy

def get_fedadam_strat(eval_strat, eta=1e-1, eta_I=1e-1, beta_1=0.9, beta_2=0.99, tau=1e-9):
    strategy = FedAdam(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=10,  # Never sample less than 10 clients for training
        min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
        min_available_clients=10,  # Wait until all 10 clients are available
        evaluate_metrics_aggregation_fn=eval_strat,  # Use the weighted average as evaluation strategy
        fit_metrics_aggregation_fn=eval_strat,  # Use the weighted average as training strategy
        eta=eta, # Server side learning rate
        eta_I=eta_I, # Client side learning rate
        beta_1=beta_1, # momentum parameter
        beta_2=beta_2, # second momentum parameter
        tau=tau # Controls the algorithms degree of adaptability
    )
    return strategy

class _CustomStrategyBase(Strategy):

    def __init__(self):
        super().__init__()

    def initialize_parameters(self, config):
        """
        Initialize parameters for the federated learning model.
        """
        return Parameters(tensors=[tf.random.normal((10, 10)) for _ in range(5)])

    def aggregate_fit(self, rnd, results, failures):
        """
        Aggregate client updates after each round of training.
        """
        weights = [res.parameters for res in results]
        avg_weights = [tf.reduce_mean(weight, axis=0) for weight in zip(*weights)]
        return Parameters(tensors=avg_weights)
    
    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggregate evaluation results.
        """
        return None

    def evaluate(self, parameters, config):
        """
        Override to implement custom evaluation logic.
        """
        return 0.4, 0.85  # Dummy evaluation result

    def configure_fit(self, rnd, parameters, client_manager, config):
        """
        Configure the fit operation for clients.
        You can modify the config here before sending it to clients.
        """
        # For example, configure learning rate or batch size for each client
        config["learning_rate"] = 0.01
        config["batch_size"] = 32
        return config  # This modified config will be passed to clients

    def configure_evaluate(self, rnd, parameters, client_manager, config):
        """
        Configure the evaluation operation for clients.
        You can modify the config here before sending it to clients.
        """
        # For example, configure validation data size or other evaluation parameters
        config["validation_data_size"] = 1000
        return config  # This modified config will be passed to clients
