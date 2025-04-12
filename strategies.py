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


