class Config:
    NUM_CLIENTS = 10
    NUM_ROUNDS = 50
    BATCH_SIZE = 32
    MODEL_NAME = "net"
    DATASET = 'cifar100' # Literal['cifar10', 'cifar100', 'mnist']
    PARTITIONER_TYPE = 'dirichlet' # Literal['uniform', 'dirichlet']
    DIRILECT_ALPHA = 0.5 # parameter when using dirichlet (means how close to iid it is)
    OPTIMIZER='r_adam'
    LOCAL_EPOCHS=2
