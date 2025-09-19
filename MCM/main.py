import argparse
import csv
import torch
import numpy as np
from Trainer import Trainer
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    parser = argparse.ArgumentParser(description='Disent-AD Training Parameters')

    # Dataset related parameters
    parser.add_argument('--dataset_name', type=str, default='vowels',
                        help='The name of the dataset')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='The directory of the dataset')
    parser.add_argument('--epochs', type=int, default=200,
                        help='The number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='The learning rate')
    parser.add_argument('--sche_gamma', type=float, default=0.98,
                        help='The gamma for the scheduler')
    parser.add_argument('--mask_num', type=int, default=15,
                        help='The number of masks')
    parser.add_argument('--lambda', type=float, default=5,
                        help='The lambda for the loss')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to use')
    parser.add_argument('--runs', type=int, default=5,
                        help='The number of runs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='The batch size')
    parser.add_argument('--en_nlayers', type=int, default=3,
                        help='The number of encoder layers')
    parser.add_argument('--de_nlayers', type=int, default=3,
                        help='The number of decoder layers')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='The hidden dimension')
    parser.add_argument('--z_dim', type=int, default=128,
                        help='The dimension of the latent space')
    parser.add_argument('--mask_nlayers', type=int, default=3,
                        help='The number of mask layers')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='The random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='The number of workers')
    return parser.parse_args()


model_config = {
    'dataset_name': 'vowels',
    'epochs': 200,
    'learning_rate': 0.05,
    'sche_gamma': 0.98,
    'mask_num': 15,
    'lambda': 5,
    'device': 'cuda:0',
    'data_dir': './data',
    'runs': 5,
    'batch_size': 512,
    'en_nlayers': 3,
    'de_nlayers': 3,
    'hidden_dim': 256,
    'z_dim': 128,
    'mask_nlayers': 3,
    'random_seed': 42,
    'num_workers': 0
}

if __name__ == "__main__":
    args = parse_args()
    model_config = vars(args)
    torch.manual_seed(model_config['random_seed'])
    torch.cuda.manual_seed(model_config['random_seed'])
    np.random.seed(model_config['random_seed'])
    if model_config['num_workers'] > 0:
        torch.multiprocessing.set_start_method('spawn')
    result = []
    runs = model_config['runs']
    mse_rauc, mse_ap, mse_f1 = np.zeros(runs), np.zeros(runs), np.zeros(runs)
    for i in range(runs):
        trainer = Trainer(run=i, model_config=model_config)
        mse_rauc[i], mse_ap[i], mse_f1[i] = trainer.training(model_config['epochs'])
        # trainer.evaluate(mse_rauc, mse_ap, mse_f1)
    mean_mse_auc, mean_mse_pr, mean_mse_f1 = np.mean(mse_rauc), np.mean(mse_ap), np.mean(mse_f1)

    print('##########################################################################')
    print("mse: average AUC-ROC: %.4f  average AUC-PR: %.4f"
          % (mean_mse_auc, mean_mse_pr))
    print("mse: average f1: %.4f" % (mean_mse_f1))
    results_name = './results/' + model_config['dataset_name'] + '.txt'
