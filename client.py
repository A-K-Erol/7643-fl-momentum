from flwr.client import Client, NumPyClient
from flwr.common import Context
from models import get_parameters, set_parameters, train, test, get_model
from data import load_datasets
from config import Config
import torch
import torch.nn as nn
import csv
metrics : dict[str, float] = {}

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, testloader, partition_id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.partition_id = partition_id

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        metrics = compute_metrics(self.net, self.valloader)
        return get_parameters(self.net), len(self.trainloader), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        # loss, accuracy = test(self.net, self.valloader)
        metrics = compute_metrics(self.net, self.valloader)
        
        with open(f"metrics_client_{self.partition_id}_sgd1.csv", "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:  # If file is empty, write header
                writer.writerow(["client_id", "accuracy", "f1"])
            writer.writerow([self.partition_id, metrics["accuracy"], metrics["f1_score"]])
                        
        return float(metrics['loss']), len(self.valloader), metrics
    

def client_fn(context: Context) -> Client:
    # Load model
    net = get_model(Config.MODEL_NAME)
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, test_loader = load_datasets(partition_id=partition_id)
    
    return FlowerClient(net, trainloader, valloader, test_loader, partition_id).to_client()


def compute_metrics(model, testloader):
    """
    Comprehensive metrics computation function.
    
    Args:
        model: PyTorch model to evaluate
        testloader: DataLoader for test dataset
    
    Returns:
        Dictionary of performance metrics
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    # Prepare for multi-class metrics
    num_classes = len(testloader.dataset.features['label'].names)
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in testloader:
            inputs = batch['img']
            labels = batch['label']
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Predictions
            _, predicted = torch.max(outputs, 1)
            
            # Update metrics
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            
            # Update confusion matrix
            for t, p in zip(labels, predicted):
                confusion_matrix[t, p] += 1
    
    # Compute accuracy
    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(testloader)
    
    # Compute precision, recall, F1 for each class
    precision = []
    recall = []
    f1_scores = []
    
    for i in range(num_classes):
        tp = confusion_matrix[i, i].item()
        fp = confusion_matrix[:, i].sum().item() - tp
        fn = confusion_matrix[i, :].sum().item() - tp
        
        # Compute precision
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        # Compute recall
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Compute F1 score
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        precision.append(prec)
        recall.append(rec)
        f1_scores.append(f1)
    
    # Macro-averaged metrics
    macro_precision = sum(precision) / num_classes
    macro_recall = sum(recall) / num_classes
    macro_f1 = sum(f1_scores) / num_classes
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'precision': macro_precision,
        'recall': macro_recall,
        'f1_score': macro_f1,
        'num_examples': total_samples
    }