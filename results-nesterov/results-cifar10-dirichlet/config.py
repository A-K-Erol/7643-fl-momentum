class Config:
    NUM_CLIENTS = 10
    NUM_ROUNDS = 50
    BATCH_SIZE = 32
    MODEL_NAME = "net"
    DATASET = 'cifar10' # Literal['cifar10', 'cifar100', 'mnist']
    PARTITIONER_TYPE = 'dirichlet' # Literal['uniform', 'dirichlet']
    DIRILECT_ALPHA = 0.5 # parameter when using dirichlet (means how close to iid it is)
    OPTIMIZER='nesterov'
    MOMENTUM=0.9
    LEARNING_RATE=0.06
