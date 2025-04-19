from flwr.server.strategy import FedAvg
from config import Config

def get_fedavg_strat(eval_strat):
    strategy = FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=Config.NUM_CLIENTS,  # Never sample less than 10 clients for training
        min_evaluate_clients=Config.NUM_CLIENTS,  # Never sample less than 5 clients for evaluation
        min_available_clients=Config.NUM_CLIENTS,  # Wait until all 10 clients are available
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
    """
    Base class which is to be extended for each custom strategy.
    """

    def __init__(
            self,         
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2
    ):
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

    def initialize_parameters(self, config):
        """
        Initialize parameters for the federated learning model.
        """
        # TODO: Change based on how to get number of parameters in net
        return Parameters(tensors=[tf.random.normal((10, 10)) for _ in range(5)])

    def aggregate_fit(self, rnd, results, failures):
        """
        Aggregate client updates after each round of training.
        """
        # The default is weighted average
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated
    
    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggregate evaluation results.
        """
        # The default is weighted average
        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(self, parameters, config):
        """
        Override to implement custom evaluation logic.
        """
        # Default is not to override
        return None

    def configure_fit(self, rnd, parameters, client_manager):
        """
        Configure the fit operation for clients.
        """
        # Return same config to all available clients
        standard_config = {"lr": 1e-3}
        available_clients = [client for client in client_manager.clients if client.is_available]
        fit_configs = [(client, FitIns(parameters, standard_config)) for client in available_clients]
        return fit_configs


    def configure_evaluate(self, rnd, parameters, client_manager):
        """
        Configure the evaluation operation for clients.
        """
        if self.fraction_evaluate == 0.0:
            return []
        
        # Return same config to all available clients
        config = {}
        available_clients = [client for client in client_manager.clients if client.is_available]
        eval_configs = [(client, EvaluateIns(parameters, config)) for client in available_clients]
        return eval_configs
    

class StrategyDefault(_CustomStrategyBase):
    def __init__(self, fraction_fit = 1, fraction_evaluate = 1, min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2):
        super().__init__(fraction_fit, fraction_evaluate, min_fit_clients, min_evaluate_clients, min_available_clients)
        
    def __repr__(self):
        return "StrategyDefault"
