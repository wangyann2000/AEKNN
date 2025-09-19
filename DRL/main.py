import csv
import torch
import numpy as np
import argparse
import os
from scipy import io
import importlib
from sklearn.cluster import KMeans
import glob
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

mat_files = glob.glob(os.path.join('../data', '*.mat'))
mat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in mat_files]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='pima')
    parser.add_argument('--model_type', type=str, default='DRL')
    parser.add_argument('--preprocess', type=str, default='standard')
    parser.add_argument('--diversity', type=str, default='True')
    parser.add_argument('--plearn', type=str, default='False')
    parser.add_argument('--input_info', type=str, default='True')
    parser.add_argument('--input_info_ratio', type=float, default=0.1)
    parser.add_argument('--cl', type=str, default='True')
    parser.add_argument('--cl_ratio', type=float, default=0.06)
    parser.add_argument('--basis_vector_num', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--runs', type=int, default=5)
    # parser.add_argument('--batch_size', type=int)
    args = parser.parse_args()

    diversity = True if args.diversity == 'True' else False
    plearn = True if args.plearn == 'True' else False
    input_info = True if args.input_info == 'True' else False
    cl = True if args.cl == 'True' else False

    dict_to_import = 'model_config_'+args.model_type


    model_config = {'epochs': args.epoch, 'learning_rate': 0.05, 'sche_gamma': 0.98, 'device': 'cuda:0',
                    'data_dir': '../data', 'runs': 1, 'batch_size': 512, 'en_nlayers': 3, 'de_nlayers': 3,
                    'hidden_dim': 128, 'random_seed': args.seed, 'num_workers': 0, 'preprocess': args.preprocess,
                    'diversity': diversity, 'plearn': plearn, 'input_info': input_info,
                    'input_info_ratio': args.input_info_ratio, 'cl': cl, 'cl_ratio': args.cl_ratio}

    torch.manual_seed(model_config['random_seed'])
    torch.cuda.manual_seed(model_config['random_seed'])
    np.random.seed(model_config['random_seed'])
    if model_config['num_workers'] > 0:
        torch.multiprocessing.set_start_method('spawn')

    path = os.path.join(model_config['data_dir'], args.dataname + '.mat')
    data = io.loadmat(path)
    samples = data['X']
    model_config['dataset_name'] = args.dataname
    model_config['data_dim'] = samples.shape[-1]

    if args.model_type == 'DRL':
        from Trainer import Trainer
        model_config['basis_vector_num'] = args.basis_vector_num

    model_config['runs'] = args.runs
    runs = model_config['runs']
    mse_rauc, mse_ap, mse_f1 = np.zeros(runs), np.zeros(runs), np.zeros(runs)

    for i in range(runs):
        trainer = Trainer(run=i, model_config=model_config)
        mse_rauc[i] , mse_ap[i] , mse_f1[i] = trainer.training(model_config['epochs'])
    mean_mse_auc , mean_mse_pr , mean_mse_f1 = np.mean(mse_rauc), np.mean(mse_ap), np.mean(mse_f1)

    print('##########################################################################')
    print("AUC-ROC: %.4f  AUC-PR: %.4f"
          % (mean_mse_auc, mean_mse_pr))
    print("f1: %.4f" % (mean_mse_f1))

