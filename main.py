import torch
import numpy as np
import argparse
from AEKNN import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='AEKNN Training Parameters')

    # Dataset related parameters
    parser.add_argument('--dataset_name', type=str, default='pima',
                        help='The name of the dataset')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='The directory of the dataset')

    # Training related parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='The number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='The learning rate')
    parser.add_argument('--sche_gamma', type=float, default=0.98,
                        help='The gamma value of the learning rate scheduler')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='The batch size')
    parser.add_argument('--runs', type=int, default=5,
                        help='The number of runs')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='The random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='The number of workers for the dataloader')

    # Model architecture parameters
    parser.add_argument('--en_nlayers', type=int, default=3,
                        help='The number of layers in the encoder')
    parser.add_argument('--de_nlayers', type=int, default=3,
                        help='The number of layers in the decoder')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='The dimension of the hidden layer')
    parser.add_argument('--z_dim', type=int, default=128,
                        help='The dimension of the latent space')

    # Device parameters
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device for training')

    parser.add_argument('--k_neighbors', type=int, default=5,
                        help='Number of nearest neighbors for KNN-based anomaly scoring')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Convert args to dictionary format to maintain compatibility with original code
    model_config = vars(args)

    torch.manual_seed(model_config['random_seed'])
    torch.cuda.manual_seed(model_config['random_seed'])
    np.random.seed(model_config['random_seed'])
    if model_config['num_workers'] > 0:
        torch.multiprocessing.set_start_method('spawn')
    result = []
    runs = model_config['runs']
    mse_roc, mse_pr, mse_f1 = np.zeros(runs), np.zeros(runs), np.zeros(runs)
    for i in range(runs):
        trainer = Trainer(run=i, model_config=model_config)
        logger, mse_roc[i], mse_pr[i], mse_f1[i] = trainer.training(model_config['epochs'])
    mean_mse_auc, mean_mse_pr, mean_mse_f1 = np.mean(mse_roc), np.mean(mse_pr), np.mean(mse_f1)

    logger.info(f"##########################################################################")
    logger.info(
        f"5 runs finished. Average AUC-ROC: {mean_mse_auc:.4f} Average AUC-PR: {mean_mse_pr:.4f} Average F1-Score:{mean_mse_f1:.4f}")
