from data_utils import load_data
from utils import get_parameters
from models import Autoencoder
from utils import set_parameters, test
import flwr as fl

NUM_CLIENTS = 10
train_path = "IDS2018_train.npy"
test_path = "IDS2018_test.npy"

params = get_parameters([net() for net in Autoencoder])

def fit_config(rnd: int):
    """Return training configuration dict for each round.
    
    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "current_round": rnd,  # The current round of federated learning
        "local_epochs": 3
    }
    return config

_, _, valid_loaders = load_data(train_path, test_path, NUM_CLIENTS)
AE = [net() for net in Autoencoder]

def evaluate(
    weights: fl.common.Weights,
):
    set_parameters(AE, weights)  # Update model with the latest parameters
    loss, accuracy = test(AE[0], valid_loaders[9])
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.3,
    fraction_eval=0.3,
    min_fit_clients=3,
    min_eval_clients=3,
    min_available_clients=3,
    initial_parameters=fl.common.weights_to_parameters(params),
    eval_fn=evaluate,
    on_fit_config_fn=fit_config,  # Pass the fit_config function
)

fl.server.start_server(
    server_address="[::]:8080",
    config={
    "num_rounds":20
    }, 
    strategy=strategy
)
