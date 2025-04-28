class Config:
    NUM_CLIENTS = 2
    NUM_ROUNDS = 1
    BATCH_SIZE = 32
    MODEL_NAME = "net"
    DATASET = 'cifar100' # Literal['cifar10', 'cifar100', 'mnist']
    PARTITIONER_TYPE = 'uniform' # Literal['uniform', 'dirichlet']
    DIRILECT_ALPHA = 0.5 # parameter when using dirichlet (means how close to iid it is)
    OPTIMIZER='accsgd' # 'sgd', 'polyak', 'nesterov', 'rectified_adam', 'qhm', 'accsgd'
    LOCAL_EPOCHS=2
    LEARNING_RATE=0.6
    MOMENTUM=0.9