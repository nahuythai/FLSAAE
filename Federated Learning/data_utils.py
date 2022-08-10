import numpy as np
from sklearn import preprocessing
import torch
from sklearn.model_selection import train_test_split
import torch.utils.data as data

def load_datasets(train_path, test_path):
    # Download and transform CIFAR-10 (train and test)
    train = np.load(train_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    print("Train records: ", train.shape)
    print("Test records: ", test.shape)
    m, n = train.shape
    testX = test[:, 0:n-1]
    testY = test[:, n-1]
    testY = testY.astype(int)
    trainX = train[:, 0:n-1]
    trainY = train[:, n-1]
    trainY = trainY.astype(int)
    scaler = preprocessing.MinMaxScaler()
    trainX = scaler.fit_transform(trainX)
    scaler = preprocessing.MinMaxScaler()
    testX = scaler.fit_transform(testX)
    scaler = preprocessing.MinMaxScaler()
    trainX = scaler.fit_transform(trainX)
    testX = torch.from_numpy(testX)
    testY = torch.from_numpy(testY)

    ul_dataX, l_dataX, ul_dataY, l_dataY = train_test_split(trainX, trainY, test_size=0.1, random_state=42)
    ul_dataY = torch.from_numpy(np.array([-1] * ul_dataY.shape[0]))
    ul_dataX = torch.from_numpy(ul_dataX)
    l_dataX = torch.from_numpy(l_dataX)
    l_dataY = torch.from_numpy(l_dataY)

    trainset_labeled = data.TensorDataset(l_dataX.float(), l_dataY)
    trainset_unlabeled = data.TensorDataset(ul_dataX.float(), ul_dataY)
    testset = data.TensorDataset(testX.float(), testY)
    return trainset_labeled, trainset_unlabeled, testset


def split_data(trainset_labeled, trainset_unlabeled, testset, num_clients, batch_size):
    # Split training set into `num_clients` partitions to simulate different local datasets
    trainset_labeled_partition_size = len(trainset_labeled) // num_clients
    trainset_labeled = data.TensorDataset(trainset_labeled[:trainset_labeled_partition_size * num_clients][0],trainset_labeled[:trainset_labeled_partition_size * num_clients][1])
    trainset_labeled = data.random_split(trainset_labeled, [trainset_labeled_partition_size] * num_clients, torch.Generator().manual_seed(42))
    # Split each partition into train/val and create DataLoader
    trainset_unlabeled_partition_size = len(trainset_unlabeled) // num_clients
    trainset_unlabeled = data.TensorDataset(trainset_unlabeled[:trainset_unlabeled_partition_size * num_clients][0],trainset_unlabeled[:trainset_unlabeled_partition_size * num_clients][1])
    trainset_unlabeled = data.random_split(trainset_unlabeled, [trainset_unlabeled_partition_size] * num_clients, torch.Generator().manual_seed(42))
    # split test
    testset_partition_size = len(testset) // num_clients
    testset = data.TensorDataset(testset[:testset_partition_size * num_clients][0],testset[:testset_partition_size * num_clients][1])
    testset = data.random_split(testset, [testset_partition_size] * num_clients, torch.Generator().manual_seed(42))

    train_labeled_loaders = []
    train_unlabeled_loaders = []
    valid_loaders = []
    for ds in trainset_labeled:
        train_labeled_loaders.append(data.DataLoader(ds, batch_size, shuffle=True))
    for ds in trainset_unlabeled:
        train_unlabeled_loaders.append(data.DataLoader(ds, batch_size, shuffle=True))
    for ds in testset:
        valid_loaders.append(data.DataLoader(ds, batch_size, shuffle=True))

    return train_labeled_loaders, train_unlabeled_loaders, valid_loaders

def load_data(train_path, test_path, num_clients, batch_size=128):
    trainset_labeled, trainset_unlabeled, testset = load_datasets(train_path, test_path)
    return split_data(trainset_labeled, trainset_unlabeled, testset, num_clients, batch_size)
