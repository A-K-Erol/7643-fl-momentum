class Config:
    NUM_CLIENTS = 10
    NUM_ROUNDS = 50
    BATCH_SIZE = 32
    MODEL_NAME = "net"
    DATASET = 'cifar100' # Literal['cifar10', 'cifar100', 'mnist']
    PARTITIONER_TYPE = 'uniform' # Literal['uniform', 'dirichlet']
    DIRILECT_ALPHA = 0.5 # parameter when using dirichlet (means how close to iid it is)
    OPTIMIZER='nesterov'
    MOMENTUM=0.8
    LEARNING_RATE=0.025
