from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np

lr = 0.001
b1 = 0.5
b2 = 0.999
n_classes = 2
latent_dim = 10
Tensor = torch.FloatTensor
DEVICE = torch.device("cpu")


def get_parameters(AE):
    weights = []
    net_keys = []
    for net in AE:
        net_keys += list(net.state_dict().keys())
        for _, val in net.state_dict().items():
            weights.append(val.cpu().numpy())
    return weights


def set_parameters(AE, parameters):
    params_dict = []
    i = 0
    for net in AE:
        net_keys = list(net.state_dict().keys())
        params_dict = zip(net_keys, parameters[i:i + len(net_keys)])
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        i = i + len(net_keys)


def sample_categorical(batch_size, n_classes=2):
    '''
     Sample from a categorical distribution
     of size batch_size and # of classes n_classes
     return: torch.autograd.Variable with the sample
    '''
    cat = np.random.randint(0, 2, batch_size)
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return cat

def train(AE, train_labeled_loader,train_unlabeled_loader, epochs=1):
    # Initial
    encoder, decoder, discriminator, discriminator_cat = AE
    # loss
    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.MSELoss()
    # optimizer
    # 1) basic optimizer for G(encoder-decoder) and D(discriminator) 
    optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # 2) additional optimizer for E-semi(encoder for semi-supervised learning) and D_cat(Discriminator for category)
    optimizer_E_semi = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D_cat = torch.optim.Adam(discriminator_cat.parameters(), lr=lr, betas=(b1, b2))

    encoder.train()
    decoder.train()
    discriminator.train()
    discriminator_cat.train()
    # training
    for _ in range(epochs):
        for (x_l, idx_l), (x_u, idx_u) in zip(train_labeled_loader, train_unlabeled_loader):
            for x, idx in [(x_l, idx_l), (x_u, idx_u)]:
                if idx[0] == -1:
                    labeled = False
                else:
                    labeled = True
                
                if labeled: # supervised phase
                    optimizer_E_semi.zero_grad()
                    _, fake_cat = encoder(x)
                    class_loss = F.cross_entropy(fake_cat, idx)
                    class_loss.backward()
                    optimizer_E_semi.step()

                else: # unsupervised phase

                    valid = nn.Parameter(Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False)
                    fake = nn.Parameter(Tensor(x.shape[0], 1).fill_(0.0), requires_grad=False)

                    # 1) reconstruction + generator loss for gaussian and category
                    optimizer_G.zero_grad()
                    fake_z, fake_cat = encoder(x)
                    decoded_x = decoder(fake_z)
                    validity_fake_z = discriminator(fake_z)
                    G_loss = 0.01*adversarial_loss(validity_fake_z, valid) + 0.01*adversarial_loss(discriminator_cat(fake_cat), valid) + 0.98*reconstruction_loss(decoded_x, x)
                    G_loss.backward()
                    optimizer_G.step()

                    # 2) discriminator loss for gaussian
                    optimizer_D.zero_grad()
                    real_z = nn.Parameter(Tensor(np.random.normal(0,1,(x.shape[0], latent_dim))), requires_grad=False)
                    real_loss = adversarial_loss(discriminator(real_z), valid)
                    fake_loss = adversarial_loss(discriminator(fake_z.detach()), fake)
                    D_loss = 0.5*(real_loss + fake_loss)
                    D_loss.backward()
                    optimizer_D.step()

                    # 3) discriminator loss for category 
                    optimizer_D_cat.zero_grad()
                    real_cat = nn.Parameter(sample_categorical(x.shape[0], n_classes=n_classes), requires_grad=False)
                    real_cat_loss = adversarial_loss(discriminator_cat(real_cat), valid)
                    fake_cat_loss = adversarial_loss(discriminator_cat(fake_cat.detach()), fake)
                    D_cat_loss = 0.5*(real_cat_loss + fake_cat_loss)
                    D_cat_loss.backward()
                    optimizer_D_cat.step()


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)[1]
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy