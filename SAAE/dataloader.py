import torch
import torch.utils.data as data
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def load_data(path_train,path_test):
  print('Loading data!')
  ratio = 0.1
  train_batch_size = 64
  valid_batch_size = 64
  train = np.load(path_train, allow_pickle=True)
  test = np.load(path_test, allow_pickle=True)
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
  ul_dataX, l_dataX, ul_dataY, l_dataY = train_test_split(trainX, trainY, test_size=ratio, random_state=42)
  ul_dataY = torch.from_numpy(np.array([-1] * ul_dataY.shape[0]))
  ul_dataX = torch.from_numpy(ul_dataX)
  l_dataX = torch.from_numpy(l_dataX)
  l_dataY = torch.from_numpy(l_dataY)
  testX = torch.from_numpy(testX)
  testY = torch.from_numpy(testY)

  trainset_labeled = data.TensorDataset(l_dataX.float(), l_dataY)
  trainset_unlabeled = data.TensorDataset(ul_dataX.float(), ul_dataY)
  validset = data.TensorDataset(testX.float(), testY)
  train_labeled_loader = data.DataLoader(trainset_labeled, batch_size=train_batch_size, shuffle=True)
  train_unlabeled_loader = data.DataLoader(trainset_unlabeled, batch_size=train_batch_size, shuffle=True)
  valid_loader = data.DataLoader(validset, batch_size=valid_batch_size, shuffle=True)
  
  return train_labeled_loader,train_unlabeled_loader,valid_loader
    