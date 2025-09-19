import csv
import os
import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import copy
from process_tabular import Data_Loader
from load_dataset import DatasetBuilder
from model import DisNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def testing(model, test_loader, device):
    model.eval()
    label_score = []
    for i, sample in enumerate(test_loader):
        data = sample['data'].to(device)
        target = sample['label'].to(device)
        with torch.no_grad():
            anomaly_score = model(data)
            label_score += list(zip(target.cpu().data.numpy().tolist(), anomaly_score.cpu().data.numpy().tolist()))
    labels, scores = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    auc_roc = roc_auc_score(labels, scores)
    auc_pr = average_precision_score(labels, scores)
    # Find optimal threshold for F1 score
    thresholds = np.percentile(scores, np.linspace(0, 100, 100))
    best_f1 = 0
    best_threshold = 0

    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        f1 = f1_score(labels, predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Final predictions with best threshold
    final_predictions = (scores > best_threshold).astype(int)
    final_f1 = f1_score(labels, final_predictions)
    return auc_roc, auc_pr, final_f1


def main(args):
    tabular_loader = Data_Loader()
    train_data, test_data, test_labels = tabular_loader.get_dataset(args.dataset)
    train_data = torch.as_tensor(train_data, dtype=torch.float)
    test_data = torch.as_tensor(test_data, dtype=torch.float)
    train_dataset = DatasetBuilder(train_data, patch_size=args.n_attr, overlap=args.overlap, norm=args.norm)
    test_dataset = DatasetBuilder(test_data, label=test_labels, patch_size=args.n_attr, overlap=args.overlap,
                                  norm=args.norm)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    model = DisNet(dim=args.hidden_dim, att_dim=args.n_attr).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_auc = -np.inf
    best_ap = -np.inf
    best_model = None
    min_loss = np.inf
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        reconstruction_loss = 0
        disentangle_loss = 0
        batch_idx = -1
        for i, sample in enumerate(train_loader):
            batch_idx += 1
            data = sample['data'].to(device)
            recon_loss, dis_loss = model(data)

            reconstruction_loss += recon_loss.item()
            disentangle_loss += dis_loss.item()
            loss = recon_loss + dis_loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss = total_loss / (batch_idx + 1)
        reconstruction_loss = reconstruction_loss / (batch_idx + 1)
        disentangle_loss = disentangle_loss / (batch_idx + 1)

        if total_loss < min_loss:
            best_model = copy.deepcopy(model)
            min_loss = total_loss

        print(
            f'Epoch: {epoch}, Total Loss: {total_loss}, Reconstruction Loss: {reconstruction_loss}, Dis Loss: {disentangle_loss}')
    auc_roc, auc_pr, final_f1 = testing(model, test_loader, device)
    print(f'AUC-ROC: {auc_roc}, AP: {auc_pr}, F1: {final_f1}')
    return auc_roc, auc_pr, final_f1



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    parser = argparse.ArgumentParser(description='Tabular Training')
    parser.add_argument('--dataset', type=str, default='pima')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N', help='batch size for training')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--n_attr', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--norm', type=str, default='minmax')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=5)
    args = parser.parse_args()

    print(args)
    runs = args.runs
    mse_rauc, mse_ap, mse_f1 = np.zeros(runs), np.zeros(runs), np.zeros(runs)
    for i in range(args.runs):
        mse_rauc[i], mse_ap[i], mse_f1[i] = main(args)

    mean_mse_auc, mean_mse_pr, mean_mse_f1 = np.mean(mse_rauc), np.mean(mse_ap), np.mean(mse_f1)
    print('##########################################################################')
    print("mse: average AUC-ROC: %.4f  average AUC-PR: %.4f"
          % (mean_mse_auc, mean_mse_pr))
    print("mse: average f1: %.4f" % (mean_mse_f1))
