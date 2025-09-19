from scipy import io
import logging
import os
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.optim as optim
from Model.Model import MCM
from Model.Loss import LossFunction
from Model.Score import ScoreFunction
from utils import aucPerformance, get_logger, F1Performance
from torch.utils.data import DataLoader, TensorDataset


class Trainer(object):
    def __init__(self, run: int, model_config: dict):
        self.run = run
        self.sche_gamma = model_config['sche_gamma']
        self.device = model_config['device']
        self.learning_rate = model_config['learning_rate']
        self.loss_fuc = LossFunction(model_config).to(self.device)
        self.score_func = ScoreFunction(model_config).to(self.device)
        self.model_config = model_config
        self.setup_logger()
        self.train_loader, self.test_loader = self.get_dataloader()
        self.model = MCM(model_config).to(self.device)

    def setup_logger(self):
        """Setup logger for training process"""
        log_dir = f'./logs/{self.model_config["dataset_name"]}'
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/run_{self.run}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f'Trainer_Run_{self.run}')

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
            self.model_config['data_dim'] = samples.shape[1]
            self.logger.info(f"Loaded .mat file: {mat_path}")
        else:
            raise FileNotFoundError(f"{mat_path} not exists")

        # Separate inliers and outliers
        inliers = samples[labels == 0]
        outliers = samples[labels == 1]

        self.logger.info(f"Dataset loaded - Inliers: {len(inliers)}, Outliers: {len(outliers)}")

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

        self.logger.info(f"Train set: {len(train_data)} samples (all normal)")
        self.logger.info(f"Test set: {len(test_data)} samples ({len(test_inliers)} normal, {len(outliers)} anomalies)")

        # Apply normalization using MinMaxScaler
        self.scaler = MinMaxScaler()
        train_data_normalized = self.scaler.fit_transform(train_data)
        test_data_normalized = self.scaler.transform(test_data)

        self.logger.info("Data normalized using MinMaxScaler")

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
        self.best_auc_pr = 0
        train_logger = get_logger('train_log.log')
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")
        min_loss = 100
        for epoch in range(epochs):
            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                x_pred, z, masks = self.model(x_input)
                loss, mse, divloss = self.loss_fuc(x_input, x_pred, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t mse={:.4f}\t divloss={:.4f}\t'
            train_logger.info(info.format(epoch, loss.cpu(), mse.cpu(), divloss.cpu()))
        print("Training complete.")
        train_logger.handlers.clear()
        val_auc, val_auc_pr, val_f1 = self.validate()

        return val_auc, val_auc_pr, val_f1

    def validate(self):
        """Validate model during training"""
        self.model.eval()
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for data, labels in self.test_loader:
                data = data.to(self.device)
                x_pred, z, masks = self.model(data)
                mse_batch = self.score_func(data, x_pred)
                mse_batch = mse_batch.data.cpu()
                all_scores.extend(mse_batch.numpy())
                all_labels.extend(labels.numpy())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # Calculate AUC-ROC
        auc_roc = roc_auc_score(all_labels, all_scores)
        auc_pr = average_precision_score(all_labels, all_scores)

        # Find optimal threshold for F1 score
        thresholds = np.percentile(all_scores, np.linspace(0, 100, 100))
        best_f1 = 0
        best_threshold = 0

        for threshold in thresholds:
            predictions = (all_scores > threshold).astype(int)
            f1 = f1_score(all_labels, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Final predictions with best threshold
        final_predictions = (all_scores > best_threshold).astype(int)
        final_f1 = f1_score(all_labels, final_predictions)

        self.logger.info(f"Validation AUC-ROC: {auc_roc:.4f}")
        self.logger.info(f"Validation AUC-PR: {auc_pr:.4f}")
        self.logger.info(f"Validation F1 Score: {final_f1:.4f}")

        return auc_roc, auc_pr, final_f1