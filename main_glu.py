# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/5/20 20:49
@author: LiFan Chen
@Filename: main_glu.py
@Software: PyCharm
"""
import pickle

import torch
import numpy as np
import random
import os
import time
from model_glu import *
import timeit
import argparse
from sklearn.model_selection import KFold

import os
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'




def load_tensor(file_name, dtype):
    data_array = np.load(file_name + '.npy', allow_pickle=True)
    desired_dtype = np.float32
    converted_data = [torch.tensor(d.astype(desired_dtype)).to(device) for d in data_array]

    # converted_data = [torch.tensor(d).to(device) for d in data_array]
    return converted_data
    # return [dtype(d).to(device) for d in np.load(file_name + '.npy',allow_pickle=True)]


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)

    DATASET = "C.elegans"
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('dataset/' + DATASET + '/word2vec_30-modify-MDL-CPI/')
    compounds1 = load_tensor(dir_input + 'compounds1', dtype=torch.float32)
    compounds2 = load_tensor(dir_input + 'compounds2', torch.long)
    compounds2 = [torch.tensor(item, dtype=torch.long).clone().detach() for item in compounds2]
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.float32)
    proteins1 = load_tensor(dir_input + 'proteins1', torch.float32)
    proteins2 = load_tensor(dir_input + 'proteins2', torch.long)
    proteins2 = [torch.tensor(item, dtype=torch.long).clone().detach() for item in proteins2]
    interactions = load_tensor(dir_input + 'interactions', torch.long)
    interactions = [torch.tensor(item, dtype=torch.long).clone().detach() for item in interactions]

    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')

    global n_fingerprint,n_word

    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    dataset = list(zip(compounds1, compounds2, adjacencies, proteins1, proteins2, interactions))

    # 5flod
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)


    fold_idx = 1


    best_results = []
    for train_index, test_index in kf.split(dataset):
        print(f"=== Fold {fold_idx} ===")
        dataset_train = [dataset[i] for i in train_index]
        dataset_test = [dataset[i] for i in test_index]

        val_split = int(0.1 * len(dataset_train))
        dataset_dev = dataset_train[:val_split]
        dataset_train = dataset_train[val_split:]

        """ create model ,trainer and tester """
        protein_dim = 100
        atom_dim = 34
        hid_dim = 64
        n_layers = 3
        n_heads = 8

        gat_heads = 3
        alpha = 0.2
        radius = 2
        ngram = 3

        pf_dim = 256
        dropout = 0.1  # 0.1
        batch = 32 # 64
        lr = 1e-3 # 1e-3
        weight_decay = 1e-4 # 1e-4
        decay_interval = 5 # 5
        lr_decay = 0.5
        iteration = 40
        kernel_size = 5

        k_feature = 16
        k_dim = 16   # tensor_neurons,16

        epoch = 40

        gat = GAT(atom_dim, hid_dim,gat_heads, dropout, alpha, n_layers,device)
        cnn = TextCNN(hid_dim, hid_dim)
        inter_att = InteractionModel(hid_dim, n_heads)
        tensor_network = TensorNetworkModule(k_feature,hid_dim,k_dim)   # NTN
        decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)    # IMT
        model = Predictor(gat,cnn, decoder, inter_att,tensor_network,device,n_fingerprint,n_layers)


        model.to(device)
        trainer = Trainer(model, lr, weight_decay, batch)
        tester = Tester(model)


        max_AUC_test = 0
        best_epoch = 0
        best_precision = 0
        best_recall = 0

        start = timeit.default_timer()

        for ep in range(1, epoch + 1):
            loss_train = trainer.train(dataset_train, device)

            AUC_dev, _, _ = tester.test(dataset_dev)
            AUC_test, precision_test, recall_test = tester.test(dataset_test)
            AUCs = [epoch, time, loss_train, AUC_dev, AUC_test, precision_test, recall_test]
            tester.save_AUCs(AUCs, f"output_C.elegans/result-C.elegans//model_fold{fold_idx}.txt")
            if AUC_test > max_AUC_test:
                max_AUC_test = AUC_test
                best_precision = precision_test
                best_recall = recall_test
                best_epoch = ep
                tester.save_model(model, f"output_C.elegans/model-C.elegans//model_fold{fold_idx}.pt")

            print(
                f"Fold {fold_idx} Epoch {ep}: Loss={loss_train:.4f}, AUC_dev={AUC_dev:.4f}, AUC_test={AUC_test:.4f}, Precision={precision_test:.4f}, Recall={recall_test:.4f}")

        end = timeit.default_timer()
        elapsed = end - start

        print(f"Fold {fold_idx} best epoch: {best_epoch} with AUC_test: {max_AUC_test:.4f}")

        best_results.append({
            "fold": fold_idx,
            "best_epoch": best_epoch,
            "AUC_test": max_AUC_test,
            "Precision": best_precision,
            "Recall": best_recall,
            "time": elapsed
        })

        fold_idx += 1

    AUCs = [r["AUC_test"] for r in best_results]
    Precisions = [r["Precision"] for r in best_results]
    Recalls = [r["Recall"] for r in best_results]

    print("\n===== five-fold cross validation results =====")
    print(f"AUC_test: average={np.mean(AUCs):.4f}, SD={np.std(AUCs):.4f}")
    print(f"Precision: average={np.mean(Precisions):.4f}, SD={np.std(Precisions):.4f}")
    print(f"Recall:    average={np.mean(Recalls):.4f}, SD={np.std(Recalls):.4f}")




