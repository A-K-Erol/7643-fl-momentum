from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context, Metrics
from flwr.server.strategy import Strategy
from strategies import get_fedavg_strat
from typing import List, Tuple
from config import Config

def server_fn(context: Context) -> ServerAppComponents:
    
    """Construct components that set the ServerApp behaviour.

    You can use settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}
    
    metrics_aggregator = create_metrics_aggregator(
        metrics_to_aggregate=['accuracy', 'loss', 'precision', 'recall', 'f1_score'],
        aggregation_method='weighted_average'
    )
    # Create FedAvg strategy
    strategy = get_fedavg_strat(metrics_aggregator)

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=Config.NUM_ROUNDS)

    return ServerAppComponents(strategy=strategy, config=config)


def create_metrics_aggregator(
    metrics_to_aggregate: List[str] = ['accuracy', 'loss'],
    aggregation_method: str = 'weighted_average'
):
    """
    Create a flexible metrics aggregation function for federated learning.
    
    Args:
        metrics_to_aggregate: List of metrics to aggregate
        aggregation_method: Method of aggregation ('weighted_average', 'average')
    
    Returns:
        A metrics aggregation function
    """
    def aggregate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """
        Aggregate metrics across clients with customizable methods.
        
        Args:
            metrics: List of tuples containing (num_examples, client_metrics)
        
        Returns:
            Aggregated metrics dictionary
        """
        # Initialize aggregation results
        aggregated_results = {}
        
        for metric in metrics_to_aggregate:
            if aggregation_method == 'weighted_average':
                # Weighted average (default)
                metric_values = [num_examples * m.get(metric, 0) for num_examples, m in metrics]
                total_examples = sum(num_examples for num_examples, _ in metrics)
                
                # Avoid division by zero
                aggregated_results[metric] = (
                    sum(metric_values) / total_examples 
                    if total_examples > 0 
                    else 0
                )
            
            elif aggregation_method == 'average':
                # Simple average
                metric_values = [m.get(metric, 0) for _, m in metrics]
                aggregated_results[metric] = (
                    sum(metric_values) / len(metric_values) 
                    if len(metric_values) > 0 
                    else 0
                )
        
        # Additional aggregation metrics
        aggregated_results['num_clients'] = len(metrics)
        aggregated_results['total_examples'] = sum(num_examples for num_examples, _ in metrics)
        
        return aggregated_results
    
    return aggregate_metrics
