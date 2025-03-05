import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'


from flwr.client import ClientApp
from flwr.server import ServerApp
from client import client_fn
from server import server_fn
from flwr.simulation import run_simulation
import torch



client = ClientApp(client_fn=client_fn)
server = ServerApp(server_fn=server_fn)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
if DEVICE == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}

run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=10,
    backend_config=backend_config,
)