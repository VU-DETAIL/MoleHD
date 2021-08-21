import codecs
from SmilesPE.tokenizer import *
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import sklearn.metrics 
import imblearn
import os, argparse
from tqdm import tqdm
import sys
from datetime import datetime
import time
import copy
import random


def clean_dataset(X, Y):
    
    X_clean = []
    Y_clean = []
    
    X_bad = []
    Y_bad = []
    
    for x, y in zip(X, Y):
        try:
            splitter = dc.splits.ScaffoldSplitter()
            dataset = dc.data.NumpyDataset(X=np.array([x]), ids=[x])
            train, valid, test = splitter.split(dataset)
            X_clean.append(x)
            Y_clean.append(y)
        except:
            X_bad.append(x)
            Y_bad.append(y)
    
    return X_clean, Y_clean, X_bad, Y_bad


def train_test_split_idx(X, Y, split="scafflod"):
    
    if split.lower() == "scafflod":
        splitter = dc.splits.ScaffoldSplitter()
        dataset = dc.data.NumpyDataset(X=np.array(X), ids=X)
        train_idxs, valid_idxs, test_idxs = splitter.split(dataset, 
                                                           frac_train=0.8, 
                                                           frac_valid=0.10, 
                                                           frac_test=0.10)
        test_idxs += valid_idxs
        
    elif split.lower() == "random":
        ids = [i for i in range(len(X))]
        _, _, train_idxs, test_idxs = train_test_split(X, ids, test_size=0.2, random_state=10)
        
    elif split.lower() == "random_stratified":
        ids = [i for i in range(len(X))]
        _, _, train_idxs, test_idxs = train_test_split(X, ids, test_size=0.2, random_state=10, stratify=Y)
    else:
        print(f"Currently, we do not support {split} split. Please try one of these three [scaffold, random, random_stratified]")
    
    return (train_idxs, test_idxs)


def tokenize_data(X, num_tokens=1500):
    
    spe_vob= codecs.open('../data/SPE_ChEMBL.txt')
    spe = SPE_Tokenizer(spe_vob)

    # Build dictionary using all the data
    scopus = [] 

    for sample in X:
        sample_token = spe.tokenize(sample).split(' ')
        scopus = scopus + sample_token

    dict = Counter(scopus).most_common(num_tokens)
    dict = [x[0] for x in dict]
    
    data_tokenized = []
    for sample in X:
        sample_token = spe.tokenize(sample).split(' ')
        sample_token = [dict.index(x) if x in dict else num_tokens for x in sample_token]
        data_tokenized.append(sample_token)
    
    return data_tokenized


def get_data_HV(data_tokenized, item_mem, gramsize, max_pos=256):
    data_HV = []
    for sample_ in data_tokenized:
        sample_ = sample_[:max_pos]
        sample_ = [item_mem[i] for i in sample_]
        sample_hv = np.ones_like(sample_[0])
        # sample_hv = sum(sample_) #plain addition
        for token_idx in range(len(sample_) - gramsize + 1):
            if gramsize == 1:
                sample_hv = sample_hv + np.roll(sample_[token_idx], token_idx)
            elif gramsize == 2:
                sample_hv = sample_hv + np.multiply(sample_[token_idx], sample_[token_idx+1])
            elif gramsize == 3:
                sample_hv = sample_hv + np.multiply(sample_[token_idx], np.multiply(sample_[token_idx+1], sample_[token_idx+2]))
            else:
                print(f"Gram size {gramsize} not supported. Please try one of [1, 2, 3].")
                break

        sample_hv[sample_hv>max_pos] = max_pos
        sample_hv[sample_hv<-max_pos] = -max_pos
        data_HV.append(sample_hv)
    
    return data_HV


def retrain(assoc_mem, epochs, dim, X_tr, Y_tr):
    for epoch in range(epochs):
        for i in range(len(Y_tr)):
            dist = cdist(X_tr[i].reshape(1, dim), assoc_mem, metric='cosine')
            dist = dist[0]
            if dist[0] < dist[1]:
                pred = 0
            else:
                pred = 1

        if(pred != Y_tr[i]):
            assoc_mem[pred] -= X_tr[i]
            assoc_mem[Y_tr[i]] += X_tr[i]

    # Setting threshold
    assoc_mem[assoc_mem < -1024] = -1024
    assoc_mem[assoc_mem > 1024] = 1024
    
    return assoc_mem


def inference(assoc_mem, dim, X_te, Y_te):
    Y_pred = []
    Y_score = []
    for i in range(len(Y_te)):
        dist = cdist(X_te[i].reshape(1, dim), assoc_mem, metric='cosine')
        dist = dist[0]
        if dist[0] < dist[1]:
            pred = 0
            # Y_score.append(0)
        else:
            pred = 1
            # Y_score.append(1)
        Y_pred.append(pred)
        Y_score.append((dist[0] - dist[1] + 2)/ 4)

    # Metrics
    acc = sklearn.metrics.accuracy_score(Y_te, Y_pred)
    bacc = sklearn.metrics.balanced_accuracy_score(Y_te, Y_pred)
    f1 = sklearn.metrics.f1_score(Y_te, Y_pred)
    precision = sklearn.metrics.precision_score(Y_te, Y_pred)
    recall = sklearn.metrics.recall_score(Y_te, Y_pred)
    roc_auc = sklearn.metrics.roc_auc_score(Y_te, Y_score)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y_te, Y_score, pos_label=1)
    
    metrics_dict = {
        "accuracy": acc, 
        "bacc": bacc,
        "f1": f1, 
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "fpr": fpr, 
        "tpr": tpr, 
        "thresholds": thresholds, 
        "confusion_matrix": sklearn.metrics.confusion_matrix(Y_te, Y_pred) 
    }
    
    return metrics_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MoleHD Framework')
    parser.add_argument('--dataset_file', default='../data/clintox.csv', type=str, help="File location. Example, '../data/clintox.csv' ")
    parser.add_argument('--target', default='CT_TOX', type=str, help="Name of target column in file.")
    parser.add_argument('--mols', default='smiles', type=str, help="Name of column that contains molecules.")
    parser.add_argument('--iterations', default=1, type=int, help="Number of iterations to run the entire experiment for. Default 1")
    parser.add_argument('--num_tokens', default=1500, type=int, help="Number of tokens to be used for data tokenization. Default 1500")
    parser.add_argument('--data_split', default=0.80, type=float, help="Split fraction for training and testing set. Defualt 0.80.")
    parser.add_argument('--random_state', default=-1, type=int, help="random state for train validation split. Defualt 42.")
    parser.add_argument('--dim', default=10000, type=int, help="Dimension of hypervector. Default 10000")
    parser.add_argument('--max_pos', default=256, type=int, help="Threshold of position hypervector. Default 256")
    parser.add_argument('--gramsize', default=1, type=int, help="N-gram tokenization size. Default 1")
    parser.add_argument('--epochs', default=150, type=int, help="Number of iterations to train the model for. Default 150")

    args = parser.parse_args()

    dataset_file = args.dataset_file
    target = args.target
    mols = args.mols
    iterations= args.datiterationsaset_file
    num_tokens = args.num_tokens
    data_split = args.data_split
    random_state = args.random_state
    dim = args.dim
    max_pos = args.max_pos
    gramsize = args.gramsize
    epochs = args.epochs

    flag = False

    file_suffix = dataset_file.split(".")[-1]

    dataset = pd.read_csv(dataset_file, sep=',', header=0)
    X = list(dataset[mols].values)
    Y = list(dataset[target].values)

    X, Y, X_bad, Y_bad = clean_dataset(X, Y)

    accuracy_list = []
    auroc_list = []
    bacc_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    confusion_matrices = []
    random_states = []

    best_assoc_mem = []
    best_auroc = 0

    data_tokenized = tokenize_data(X, num_tokens=1500)  

    if random_state == -1:
        flag = True
        random_state = random.randint(1, 1000)

    for i in tqdm(range(iterations)):
        if flag:
            random_state = random.randint(1, 1000)
        random.seed(random_state)
        item_mem = np.random.choice((-1, 1), (num_tokens+1, dim))
        pos_mem = np.random.choice((-1, 1), (max_pos, dim))

        data_HV = get_data_HV(data_tokenized, item_mem, gramsize, max_pos)

        (train_idxs, test_idxs) = train_test_split_idx(X, Y, random_state, split="scafflod")

        X_tr = []
        Y_tr = []
        X_te = []
        Y_te = []

        for idx in train_idxs:
            X_tr.append(data_HV[idx])
            Y_tr.append(Y[idx])

        for idx in test_idxs:
            X_te.append(data_HV[idx])
            Y_te.append(Y[idx])

        oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority') #Oversampling by duplication
        X_tr, Y_tr = oversample.fit_resample(X_tr, Y_tr)
        X_tr = np.array(X_tr)

        assoc_mem = np.zeros((2, dim))
        for i in range(len(Y_tr)):
            assoc_mem[Y_tr[i]] += X_tr[i]

        assoc_mem = retrain(assoc_mem, epochs, dim, X_tr, Y_tr)

        metrics_dict = inference(assoc_mem, dim, X_te, Y_te)

        if metrics_dict["roc_auc"] > best_auroc:
            best_auroc = metrics_dict["roc_auc"]
            best_assoc_mem = assoc_mem

        accuracy_list.append(metrics_dict["accuracy"])
        auroc_list.append(metrics_dict["roc_auc"])
        bacc_list.append(metrics_dict["bacc"])
        f1_list.append(metrics_dict["f1"])
        precision_list.append(metrics_dict["precision"])
        recall_list.append(metrics_dict["recall"])
        confusion_matrices.append((metrics_dict["confusion_matrix"]))
        random_states.append(random_state)
    
    print()
    print("Stats corresponding to Maximum accuracy are: ")
    
    max_auroc = max(auroc_list)
    max_auroc_idx = auroc_list.index(max_auroc)
    print("Accuracy: ", accuracy_list[max_auroc_idx])
    print("Auroc: ", auroc_list[max_auroc_idx])
    print("Bacc: ", bacc_list[max_auroc_idx])
    print("F1: ", f1_list[max_auroc_idx])
    print("Precision: ", precision_list[max_auroc_idx])
    print("Recall: ", recall_list[max_auroc_idx])
    print("Confusion Matrix: ", confusion_matrices[max_auroc_idx])
    print("Random State: ", random_states[max_auroc_idx])
    
    print()
    
    print("Stats corresponding to Minimum accuracy are: ")
    
    min_auroc = min(auroc_list)
    min_auroc_idx = auroc_list.index(min_auroc)
    print("Accuracy: ", accuracy_list[min_auroc_idx])
    print("Auroc: ", auroc_list[min_auroc_idx])
    print("Bacc: ", bacc_list[min_auroc_idx])
    print("F1: ", f1_list[min_auroc_idx])
    print("Precision: ", precision_list[min_auroc_idx])
    print("Recall: ", recall_list[min_auroc_idx])
    print("Confusion Matrix: ", confusion_matrices[min_auroc_idx])
    print("Random State: ", random_states[min_auroc_idx])
    
    print()
    
    print(f"Average Stats for {iterations} iterations")
    print("Accuracy: ", sum(accuracy_list)/len(accuracy_list))
    print("Auroc: ", sum(auroc_list)/len(auroc_list))
    print("Bacc: ", sum(bacc_list)/len(bacc_list))
    print("F1: ", sum(f1_list)/len(f1_list))
    print("Precision: ", sum(precision_list)/len(precision_list))
    print("Recall: ", sum(recall_list)/len(recall_list))
    print("Random States: ", random_states)


    iterations= args.datiterationsaset_file
    num_tokens = args.num_tokens
    data_split = args.data_split
    random_state = args.random_state
    dim = args.dim
    max_pos = args.max_pos
    gramsize = args.gramsize
    epochs = args.epochs

    with open(f'./models/model_{file_suffix}_{target}_{iterations}_{num_tokens}_{data_split*100}_{random_state}_{dim}_{max_pos}_{gramsize}_{epochs}') as f:
        pickle.dump(best_assoc_mem, f)
