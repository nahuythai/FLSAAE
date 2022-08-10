import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn import metrics


# config cuda
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_categorical(batch_size, n_classes=2):
    cat = np.random.randint(0, n_classes, batch_size)
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return cat

def classification_accuracy(Q, data_loader):
    Q.eval()
    labels = []
    test_loss = 0
    correct = 0
    for _, (X, target) in enumerate(data_loader):
        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()

        labels.extend(target.data.tolist())
        # Reconstruction phase
        output = Q(X)[1]
        test_loss += F.nll_loss(output, target).item()

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    test_loss /= len(data_loader)
    return 100. * correct / len(data_loader.dataset)

def evaluate(Q, data_loader):
    Q.eval()
    labels = []
    test_loss = 0
    correct = 0
    test_labels = []
    pred_labels = []
    for _, (X, target) in enumerate(data_loader):
        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()
        labels.extend(target.data.tolist())
        # Reconstruction phase
        output = Q(X)[1]
        test_loss += F.nll_loss(output, target).item()

        pred = output.data.max(1)[1]
        test_labels.append(target.data.cpu().tolist())
        pred_labels.append(pred.cpu().tolist())

    def flatten(t):
      return [item for sublist in t for item in sublist]
    
    pred_labels = flatten(pred_labels)
    test_labels = flatten(test_labels)

    fpr, tpr, _ = metrics.roc_curve(test_labels,pred_labels)
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(test_labels,pred_labels)
    f1 = metrics.f1_score(test_labels,pred_labels)
    recall = metrics.recall_score(test_labels,pred_labels)
    prec = metrics.precision_score(test_labels,pred_labels)

    return auc, acc, f1, recall, prec