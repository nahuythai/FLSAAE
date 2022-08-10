import random
import flwr as fl
from utils import get_parameters, set_parameters, train, test, DEVICE
from data_utils import load_data
from models import Autoencoder

NUM_CLIENTS = 10
train_path = "IDS2018_train.npy"
test_path = "IDS2018_test.npy"

class AEClient(fl.client.NumPyClient):
        def __init__(self, cid, AE,
         train_labeled_loader, train_unlabeled_loader,
         valid_loader):
            self.cid = cid
            self.AE = AE
            self.train_labeled_loader = train_labeled_loader
            self.train_unlabeled_loader = train_unlabeled_loader
            self.valid_loader = valid_loader

        def get_parameters(self):
            return get_parameters(self.AE)

        def fit(self, parameters, config):
            set_parameters(self.AE, parameters)
            train(self.AE, self.train_labeled_loader, self.train_unlabeled_loader, epochs=1)
            return self.get_parameters(), len(self.train_labeled_loader), {}

        def evaluate(self, parameters, config):
            set_parameters(self.AE, parameters)
            loss, accuracy = test(self.AE[0], self.valid_loader)
            return float(loss), len(self.valid_loader), {"accuracy": float(accuracy)}

def init_client(cid):
    train_labeled_loaders, train_unlabeled_loaders, valid_loaders = load_data(train_path, test_path, NUM_CLIENTS)
    AE = [net().to(DEVICE) for net in Autoencoder]
    train_labeled_loader = train_labeled_loaders[int(cid)]
    train_unlabeled_loader = train_unlabeled_loaders[int(cid)]
    valid_loader = valid_loaders[int(cid)]
    return AEClient(cid, AE, train_labeled_loader, train_unlabeled_loader, valid_loader)

def main():
    cid = random.randrange(NUM_CLIENTS)
    client = init_client(cid)
    fl.client.start_numpy_client("[::]:8080", client)

if __name__ == "__main__":
    main()