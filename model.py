import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors


class SimpleAutoencoder(nn.Module):
    def __init__(self, model_config):
        super(SimpleAutoencoder, self).__init__()
        self.data_dim = model_config['data_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.z_dim = model_config['z_dim']
        self.en_nlayers = model_config['en_nlayers']
        self.de_nlayers = model_config['de_nlayers']
        self.device = model_config['device']
        self.k_neighbors = model_config.get('k_neighbors', 5)  # K for KNN

        # Original feature encoder
        encoder_layers = []
        encoder_dim = self.data_dim
        for _ in range(self.en_nlayers - 1):
            encoder_layers.append(nn.Linear(encoder_dim, self.hidden_dim, bias=False))
            encoder_layers.append(nn.BatchNorm1d(self.hidden_dim))
            encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_layers.append(nn.Dropout(0.1))
            encoder_dim = self.hidden_dim
        encoder_layers.append(nn.Linear(encoder_dim, self.z_dim, bias=False))
        self.encoder = nn.Sequential(*encoder_layers)

        # Original feature decoder
        decoder_layers = []
        decoder_dim = self.z_dim
        for _ in range(self.de_nlayers - 1):
            decoder_layers.append(nn.Linear(decoder_dim, self.hidden_dim, bias=False))
            decoder_layers.append(nn.BatchNorm1d(self.hidden_dim))
            decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            decoder_layers.append(nn.Dropout(0.1))
            decoder_dim = self.hidden_dim
        decoder_layers.append(nn.Linear(decoder_dim, self.data_dim, bias=False))
        self.decoder = nn.Sequential(*decoder_layers)

        # Store training data for KNN computation
        self.train_features = None
        self.knn_model = None

    def store_training_data(self, train_loader):
        """Store training data and their latent representations for KNN computation"""
        self.eval()
        train_features_list = []
        train_recon_errors_list = []

        with torch.no_grad():
            for data, _ in train_loader:
                data = data.to(self.device)

                outputs = self.forward(data)

                recon_error = F.mse_loss(outputs['x_recon'], data, reduction='none').mean(dim=1)
                train_recon_errors_list.append(recon_error.cpu().numpy())

                concat_features = torch.cat([data, outputs['z'], outputs['x_recon']], dim=1)
                train_features_list.append(concat_features.cpu().numpy())

        if train_features_list:
            self.train_features = np.concatenate(train_features_list, axis=0)

            # Initialize KNN models
            self.knn_model = NearestNeighbors(n_neighbors=self.k_neighbors, metric='euclidean')
            self.knn_model.fit(self.train_features)

            print(f"Stored {self.train_features.shape[0]} normal training samples for KNN")

    def forward(self, x_input):
        """Forward pass through the autoencoder"""
        # Encode to latent space
        z = self.encoder(x_input)

        # Decode back to original space
        x_recon = self.decoder(z)

        return {
            'x_recon': x_recon,
            'z': z
        }

    def compute_losses(self, x_input, outputs):
        """Compute reconstruction loss only"""
        # Reconstruction loss
        loss_recon = F.mse_loss(outputs['x_recon'], x_input)

        return {
            'total_loss': loss_recon,
            'loss_original_recon': loss_recon,
        }

    def compute_knn_distances(self, x_input):
        """Compute average distances to K nearest neighbors"""
        batch_size = x_input.shape[0]
        if self.knn_model is None:
            return torch.zeros(batch_size).to(self.device)

        with torch.no_grad():
            outputs = self.forward(x_input)

            concat_features = torch.cat([x_input, outputs['z'], outputs['x_recon']], dim=1)
            concat_np = concat_features.cpu().numpy()

            distances, _ = self.knn_model.kneighbors(concat_np)
            knn_distances = torch.tensor(distances.mean(axis=1)).to(self.device)

        return knn_distances

    def compute_anomaly_score(self, x_input):
        """Compute anomaly score combining reconstruction loss and KNN distances"""
        with torch.no_grad():
            outputs = self.forward(x_input)

            # 1. Reconstruction loss
            recon_error = F.mse_loss(outputs['x_recon'], x_input, reduction='none').mean(dim=1)

            # 2. KNN distances in original and latent space
            knn_distances = self.compute_knn_distances(x_input)

            anomaly_score = knn_distances * recon_error

        return anomaly_score

# For backward compatibility, create an alias
DualChannelAutoencoder = SimpleAutoencoder