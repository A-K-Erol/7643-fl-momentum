class Config:
    NUM_CLIENTS = 5
    NUM_ROUNDS = 5
    BATCH_SIZE = 32
    MODEL_NAME = "net"
    DATASET = 'cifar10' # Literal['cifar10', 'cifar100', 'mnist']
    PARTITIONER_TYPE = 'uniform' # Literal['uniform', 'dirichlet', 'label_distribution' (unimplemented)]
    DIRILECT_ALPHA = 0.5 # parameter when using dirichlet (means how close to iid it is)
