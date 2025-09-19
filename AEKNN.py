from scipy import io
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm
import logging
import os
from model import SimpleAutoencoder


class Trainer:
    def __init__(self, run, model_config):
        self.run = run
        self.model_config = model_config
        self.device = torch.device(model_config['device'] if torch.cuda.is_available() else 'cpu')

        # Setup logger
        self.setup_logger()
        self.logger.info(f"Starting run {run + 1}")
        self.logger.info(f"Using device: {self.device}")

        # Log model configuration
        self.logger.info(f"K neighbors for KNN: {model_config.get('k_neighbors', 5)}")

        # Get dataloader
        self.train_loader, self.test_loader = self.get_dataloader()

        # Initialize model
        self.model = SimpleAutoencoder(model_config).to(self.device)

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=model_config['learning_rate']
        )

        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=model_config['sche_gamma']
        )

        # Loss weights (simplified for autoencoder only)
        self.loss_weights = {
            'original_recon': 1.0
        }

        # Best model tracking
        self.best_auc_roc = 0
        self.best_auc_pr = 0
        self.best_f1 = 0
        self.best_epoch = 0

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
        scaler = MinMaxScaler()
        train_data_normalized = scaler.fit_transform(train_data)
        test_data_normalized = scaler.transform(test_data)

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

    def setup_logger(self):
        """Setup logger for the training process"""
        log_dir = f'./logs/{self.model_config["dataset_name"]}'
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/{self.model_config["dataset_name"]}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f'Trainer_Run_{self.run}')

    def training(self, epochs):
        """Main training loop"""
        self.logger.info("Starting training...")

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = {
                'total_loss': 0,
                'loss_original_recon': 0
            }

            # Training loop with progress bar
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
            for batch_idx, (data, label) in enumerate(pbar):
                data = data.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(data)

                # Compute losses (including MLM if enabled)
                losses = self.model.compute_losses(data, outputs)

                # Backward pass
                losses['total_loss'].backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # Update epoch losses
                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key] += losses[key].item()

                # Update progress bar
                postfix_dict = {
                    'Total': f"{losses['total_loss'].item():.4f}",
                    'Recon': f"{losses['loss_original_recon'].item():.4f}"
                }
                pbar.set_postfix(postfix_dict)

            # Average losses
            num_batches = len(self.train_loader)
            for key in epoch_losses:
                epoch_losses[key] /= num_batches

            # Log epoch results
            log_msg = (f"Epoch {epoch + 1}/{epochs} - "
                       f"Total Loss: {epoch_losses['total_loss']:.4f}, "
                       f"Reconstruction Loss: {epoch_losses['loss_original_recon']:.4f}")

            self.logger.info(log_msg)

            # Learning rate scheduling
            self.scheduler.step()

        # Store training data for KNN computation
        self.logger.info("Storing training data for KNN computation...")
        val_auc, val_auc_pr, val_f1 = self.validate()

        self.logger.info(f"Training completed. AUC-ROC: {val_auc:.4f} AUC-PR: {val_auc_pr:.4f} at epoch {epoch + 1}")
        # self.save_model()
        return self.logger, val_auc, val_auc_pr, val_f1

    def validate(self):
        """Validate model during training"""
        self.model.eval()
        all_scores = []
        all_labels = []

        # Ensure training data is stored for KNN
        self.logger.info("Storing training data for KNN computation...")
        self.model.store_training_data(self.train_loader)

        with torch.no_grad():
            for data, labels in self.test_loader:
                data = data.to(self.device)
                anomaly_scores = self.model.compute_anomaly_score(data)
                all_scores.extend(anomaly_scores.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # Calculate AUC-ROC
        auc_roc = roc_auc_score(all_labels, all_scores)
        auc_pr = average_precision_score(all_labels, all_scores)

        # Find the optimal threshold for F1 score
        thresholds = np.percentile(all_scores, np.linspace(0, 100, 100))
        best_f1 = 0
        best_threshold = 0

        for threshold in thresholds:
            predictions = (all_scores > threshold).astype(int)
            f1 = f1_score(all_labels, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Final predictions with the best threshold
        final_predictions = (all_scores > best_threshold).astype(int)
        final_f1 = f1_score(all_labels, final_predictions)

        self.logger.info(f"Validation AUC-ROC: {auc_roc:.4f}")
        self.logger.info(f"Validation AUC-PR: {auc_pr:.4f}")
        self.logger.info(f"Validation F1 Score: {final_f1:.4f}")

        return auc_roc, auc_pr, final_f1