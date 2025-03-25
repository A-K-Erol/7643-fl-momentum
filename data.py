import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from config import Config
from flwr_datasets.partitioner import (
    DirichletPartitioner, 
    IidPartitioner, 
    DistributionPartitioner
)

def load_datasets(
    partition_id: int
):
    """
    Load federated datasets with configurable partitioning strategies
    
    Args:
        dataset: Name of the dataset to load
        partition_id: Specific partition to load
        num_clients: Total number of clients
        batch_size: Batch size for DataLoaders
        partitioner_type: Type of partitioning strategy
        partitioner_config: Additional configuration for partitioners
    """
    
    if Config.PARTITIONER_TYPE == 'uniform':
        partitioner = IidPartitioner(Config.NUM_CLIENTS)
    elif Config.PARTITIONER_TYPE == 'dirichlet':
        partitioner = DirichletPartitioner(alpha=Config.DIRILECT_ALPHA, num_partitions=Config.NUM_CLIENTS, partition_by="label")
    elif Config.PARTITIONER_TYPE == 'label_distribution':
        partitioner = DistributionPartitioner(num_partitions=Config.NUM_CLIENTS, distribution_array=Config.get('distribution'))
    else:
        raise ValueError(f"Unsupported partitioner type: {Config.PARTITIONER_TYPE}")

    partitioners = {'train': partitioner}

    # Dataset-specific normalization
    normalization_params = {
        'cifar10': {
            'mean': (0.5, 0.5, 0.5),
            'std': (0.5, 0.5, 0.5)
        },
        'cifar100': {
            'mean': (0.5071, 0.4867, 0.4408),
            'std': (0.2675, 0.2565, 0.2761)
        },
        'mnist': {
            'mean': (0.1307,),
            'std': (0.3081,)
        }
    }

    # Create Federated Dataset
    fds = FederatedDataset(dataset=Config.DATASET, partitioners=partitioners)
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # Prepare transforms based on dataset
    if Config.DATASET in ['cifar10', 'cifar100']:
        pytorch_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=normalization_params[Config.DATASET]['mean'], 
                std=normalization_params[Config.DATASET]['std']
            )
        ])
    elif Config.DATASET == 'mnist':
        pytorch_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=normalization_params['mnist']['mean'], 
                std=normalization_params['mnist']['std']
            )
        ])

    # Apply transforms
    def apply_transforms(batch):
        image_name = "img" if Config.DATASET != "mnist" else "image"
        batch[image_name] = [pytorch_transforms(img) for img in batch[image_name]]
        return batch
        

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    
    # Create DataLoaders
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=Config.BATCH_SIZE, shuffle=True
    )

    valloader = DataLoader(partition_train_test["test"], batch_size=Config.BATCH_SIZE)
    
    # Prepare test set
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=Config.BATCH_SIZE)
    
    return trainloader, valloader, testloader


