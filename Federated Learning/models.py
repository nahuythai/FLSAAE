import torch.nn as nn

#######################################################
# hyperparameter setting
#######################################################
latent_dim = 10
n_classes = 2
n_features = 39

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_features, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.ReLU(True),
        )

        self.lin_D = nn.Linear(1000, latent_dim)
        self.lin_D_cat = nn.Sequential(nn.Linear(1000, n_classes),nn.Softmax())


    def forward(self, img):
        x = self.model(img)

        z_gauss = self.lin_D(x)
        z_cat = self.lin_D_cat(x)

        return z_gauss, z_cat

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.ReLU(True),
            nn.Linear(1000, n_features),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        return img_flat

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

class Discriminator_category(nn.Module):
    def __init__(self):
        super(Discriminator_category, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_classes, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

Autoencoder = Encoder, Decoder, Discriminator, Discriminator_category