class Config:
    NUM_CLIENTS = 10
    NUM_ROUNDS = 5
    BATCH_SIZE = 32
    MODEL_NAME = "net"
    DATASET = 'mnist' # Literal['cifar10', 'cifar100', 'mnist']
    PARTITIONER_TYPE = 'uniform' # Literal['uniform', 'dirichlet', 'label_distribution' (unimplemented)]
    DIRILECT_ALPHA = 0.5
