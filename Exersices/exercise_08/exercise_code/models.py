import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt


class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=50):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = None

        self.encoder = nn.Sequential(
          nn.Linear(input_size,self.hparams["n_hidden"]),
          nn.ReLU(),
          nn.Linear(self.hparams["n_hidden"],latent_dim),
          nn.ReLU() 
        )


    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=50, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.latent_dim= latent_dim
        self.output_size=output_size
        self.decoder = None


        self.decoder = nn.Sequential(
          nn.Linear(self.latent_dim, self.hparams["n_hidden"]),
          nn.ReLU(),
          nn.Linear(self.hparams["n_hidden"],self.output_size)
        )

        pass


    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

    def forward(self, x):
        reconstruction = None

        latent= self.encoder(x)
        reconstruction = self.decoder(latent)        
        
        pass

        return reconstruction

    def set_optimizer(self):

        self.optimizer = None

        self.optimizer = torch.optim.Adam(
          self.parameters(),
          lr=self.hparams["learning_rate"],
          weight_decay = self.hparams["weight_decay"]
          )


    def training_step(self, batch, loss_func):
        """
        This function is called for every batch of data during training. 
        It should return the loss for the batch.
        """
        loss = None
        self.train()
        self.optimizer.zero_grad()
        images=batch
        images=images.to(self.device)
        images=images.view(images.shape[0] , -1)
        pred=self.forward(images)
        loss=loss_func(pred,images)
        loss.backward()
        self.optimizer.step()

        return loss

    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        loss = None
        validation_loss=0

        self.eval()
        images = batch
        images=images.to(self.device)

        images=images.view(images.shape[0],-1)
        pred=self.forward(images)
        loss=loss_func(pred,images)
        validation_loss+=loss.item()

        return loss

    def getReconstructions(self, loader=None):

        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        self.hparams = hparams
        self.encoder = encoder
        self.model = nn.Identity()
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()
        
        self.model = nn.Sequential(
            nn.Linear(50, self.hparams["n_hidden"]),
            nn.ReLU(),
            nn.Linear(self.hparams["n_hidden"], self.hparams["num_classes"]),
        )

        pass

    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def set_optimizer(self):
        
        self.optimizer = None
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'], weight_decay = self.hparams['weight_decay'])

        pass

    def getAcc(self, loader=None):
        
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)
            
        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
