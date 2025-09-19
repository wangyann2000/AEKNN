import logging
import torch
import torch.optim as optim
import torch.nn.functional as F
from Model import DRL
from utils import aucPerformance, get_logger, F1Performance
import numpy as np
import os
from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


class Trainer(object):
    def __init__(self, run: int, model_config: dict):
        self.run = run
        self.sche_gamma = model_config['sche_gamma']
        self.device = model_config['device']
        self.learning_rate = model_config['learning_rate']
        self.model = DRL(model_config).to(self.device)
        self.model_config = model_config
        self.train_loader, self.test_loader = self.get_dataloader()

    def load_and_preprocess_data(self):
        """Load data once and apply preprocessing including normalization"""
        dataset_name = self.model_config['dataset_name']
        data_dir = self.model_config['data_dir']

        # Determine file extension and load data accordingly
        mat_path = os.path.join(data_dir, dataset_name + '.mat')

        if os.path.exists(mat_path):
            # Load .mat file
            data = io.loadmat(mat_path)
            samples = data['X']
            labels = ((data['y']).astype(int)).reshape(-1)
        else:
            raise FileNotFoundError(f"Neither {mat_path} exists")

        # Separate inliers and outliers
        inliers = samples[labels == 0]
        outliers = samples[labels == 1]

        # Split inliers: 50% for training, 50% for testing
        num_train_inliers = len(inliers) // 2
        shuffled_indices = np.random.permutation(len(inliers))

        train_inliers = inliers[shuffled_indices[:num_train_inliers]]
        test_inliers = inliers[shuffled_indices[num_train_inliers:]]

        # Prepare training and test sets
        train_data = train_inliers
        train_labels = np.zeros(len(train_inliers))

        test_data = np.concatenate([test_inliers, outliers], axis=0)
        test_labels = np.concatenate([
            np.zeros(len(test_inliers)),
            np.ones(len(outliers))
        ], axis=0)

        # Apply normalization using MinMaxScaler
        self.scaler = MinMaxScaler()
        train_data_normalized = self.scaler.fit_transform(train_data)
        test_data_normalized = self.scaler.transform(test_data)

        return train_data_normalized, train_labels, test_data_normalized, test_labels

    def get_dataloader(self):
        # Load and preprocess data once
        train_data, train_labels, test_data, test_labels = self.load_and_preprocess_data()

        # Create tensor datasets
        train_dataset = TensorDataset(
            torch.tensor(train_data, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.float32)
        )

        test_dataset = TensorDataset(
            torch.tensor(test_data, dtype=torch.float32),
            torch.tensor(test_labels, dtype=torch.float32)
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.model_config['batch_size'],
            num_workers=self.model_config['num_workers'],
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.model_config['batch_size'],
            shuffle=False
        )

        return train_loader, test_loader

    def training(self, epochs):
        train_logger = get_logger(f"./logs/{self.model_config['dataset_name']}_DRL_{self.run}.log")
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")
        for epoch in range(epochs):
            running_loss = 0.0
            self.best_auc_pr = 0
            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                # decomposition loss
                loss = self.model(x_input).mean()

                # alignment loss
                if self.model_config['input_info'] == True:
                    h = self.model.encoder(x_input)
                    x_tilde = self.model.decoder(h)
                    # s_loss = (1-F.cosine_similarity(x_tilde, x_input, dim=-1)).mean()
                    s_loss = F.cosine_similarity(x_tilde, x_input, dim=-1).mean() * (-1)
                    loss += self.model_config['input_info_ratio'] * s_loss

                # separation loss
                if self.model_config['cl'] == True:
                    h_ = F.softmax(self.model.phi(x_input), dim=1)
                    selected_rows = np.random.choice(h_.shape[0], int(h_.shape[0] * 0.8), replace=False)
                    h_ = h_[selected_rows]

                    matrix = h_ @ h_.T
                    mol = torch.sqrt(torch.sum(h_ ** 2, dim=-1, keepdim=True)) @ torch.sqrt(
                        torch.sum(h_.T ** 2, dim=0, keepdim=True))
                    matrix = matrix / mol
                    d_loss = ((1 - torch.eye(h_.shape[0]).cuda()) * matrix).sum() / (h_.shape[0]) / (h_.shape[0])
                    loss += self.model_config['cl_ratio'] * d_loss

                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t'
            running_loss = running_loss / len(self.train_loader)
            train_logger.info(info.format(epoch, running_loss))
        # torch.save(self.model, f"./models/{self.model_config['dataset_name']}_DRL_{self.run}.pth")
        print("Training complete.")
        train_logger.handlers.clear()
        val_auc, val_auc_pr, val_f1 = self.evaluate()

        return val_auc, val_auc_pr, val_f1

    def evaluate(self):
        # model = torch.load(f"./models/{self.model_config['dataset_name']}_DRL_{self.run}.pth")
        # model.eval()
        mse_score, test_label = [], []
        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)

            h = self.model.encoder(x_input)

            weight = F.softmax(self.model.phi(x_input), dim=1)
            h_ = weight @ self.model.basis_vector

            mse = F.mse_loss(h, h_, reduction='none')
            mse_batch = mse.mean(dim=-1, keepdim=True)
            mse_batch = mse_batch.data.cpu()
            mse_score.append(mse_batch)
            test_label.append(y_label)
        mse_score = torch.cat(mse_score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        mse_rauc, mse_ap = aucPerformance(mse_score, test_label)
        mse_f1 = F1Performance(mse_score, test_label)
        return mse_rauc, mse_ap, mse_f1