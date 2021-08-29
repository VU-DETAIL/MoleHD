from SmilesPE.tokenizer import *
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import sklearn.metrics 
import imblearn
import deepchem as dc
from deepchem.splits.splitters import ScaffoldSplitter
from tqdm.notebook import tqdm, trange
from utils import *
import pickle
import os, argparse
import random


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MoleHD Framework')
    parser.add_argument('--dataset_file', default='./data/clintox.csv', type=str, help="File location. Example, '../data/clintox.csv' ")
    parser.add_argument('--target', default='CT_TOX', type=str, help="Name of target column in file.")
    parser.add_argument('--mols', default='smiles', type=str, help="Name of column that contains molecules.")
    parser.add_argument('--num_tokens', default=1500, type=int, help="Number of tokens to be used for data tokenization. Default 1500")
    parser.add_argument('--dim', default=10000, type=int, help="Dimension of hypervector. Default 10000")
    parser.add_argument('--max_pos', default=256, type=int, help="Threshold of position hypervector. Default 256")
    parser.add_argument('--gramsize', default=2, type=int, help="N-gram tokenization size. Default 1")
    parser.add_argument('--retraining_epochs', default=150, type=int, help="Number of iterations to train the model for. Default 150")
    parser.add_argument('--iterations', default=5, type=int, help="Number of iterations to run the entire experiment for. Default 1")
    parser.add_argument('--test_size', default=20, type=int, help="Split percentage for testing set. Defualt 20.")
    parser.add_argument('--threshold', default=1024, type=int, help="Threshold to scope the associate memory. Defualt 1024.")
    parser.add_argument('--encoding_scheme', default="smiles_pretrained", type=str, help="Encoding scheme for HDC. Supported types [smiles_pretrained, atomwise, characterwise]")
    parser.add_argument('--split_type', default="random", type=str, help="Data split method. Supported types [scaffold, random, random_stratified]")   
    parser.add_argument('--version', default="v1", type=str, help="Version to be appended to file name while saving model and output.")  
    
    parser.add_argument('--epochs', default=150, type=int, help="Number of iterations to train the model for. Default 150")

    args = parser.parse_args()

    dataset_file = args.dataset_file
    target = args.target
    mols = args.mols
    num_tokens = args.num_tokens
    dim = args.dim
    max_pos = args.max_pos
    gramsize = args.gramsize
    epochs = args.retraining_epochs
    iterations = args.iterations
    test_size = args.test_size
    threshold = args.threshold
    encoding_scheme = args.encoding_scheme
    split_type = args.split_type
    version = args.version

    dataset = pd.read_csv(dataset_file, sep=',', header=0)

    X = list(dataset[mols].values)
    Y = list(dataset[target].values)

    X, Y, X_bad, Y_bad = clean_dataset(X, Y)

    print(len(X), len(Y))

    accuracy_list = []
    auroc_list = []
    bacc_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    confusion_matrices = []

    metrics_dict = dict()
    metrics_dict["accuracy_list"] = list()
    metrics_dict["auroc_list"] = list()
    metrics_dict["bacc_list"] = list()
    metrics_dict["f1_list"] = list()
    metrics_dict["precision_list"] = list()
    metrics_dict["recall_list"] = list()
    metrics_dict["confusion_matrices"] = list()
    metrics_dict["random_states"] = list()


    max_assoc_mem = []
    max_auroc = 0

    for iteration in tqdm(range(iterations)):
        
        random_state = random.randint(0, 1000)
        print(f"----------- Iteration {iteration+1} ------------")
        
        if encoding_scheme.lower() == "smiles_pretrained":
            data_tokenized = data_tokenize_smiles_pretrained(X, num_tokens=num_tokens)
        elif encoding_scheme.lower() == "atomwise":
            data_tokenized = data_tokenize_atomwise(X, num_tokens=num_tokens)
        elif encoding_scheme.lower() == "characterwise":
            data_tokenized = data_tokenize_characterwise(X, num_tokens=num_tokens)
        else:
            print(f"MoleHD currently do not support {encoding_scheme} encoding scheme. Please try of of the 3 encoding schemes [scaffold, random, random_stratified]")
        
        data_HV = create_associative_memory(data_tokenized, gramsize=gramsize, num_tokens=num_tokens, dim=dim, max_pos=max_pos, random_state=10)
        
        if split_type.lower() == "scaffold":
            X_tr, X_te, Y_tr, Y_te = train_test_split_scaffold(X, Y, data_HV, test_size=test_size/100, random_state=random_state)
        elif split_type.lower() == "random":
            X_tr, X_te, Y_tr, Y_te = train_test_split(data_HV, Y, test_size=test_size/100, random_state=random_state)
        elif split_type.lower() == "random_stratified":
            X_tr, X_te, Y_tr, Y_te = train_test_split(data_HV, Y, test_size=test_size/100, random_state=random_state, stratify=Y)
        else:
            print(f"MoleHD currently do not support {split_type} split type. Please try of of the 3 encoding schemes [scaffold, random, random_stratified]")
            
        
        oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority') #Oversampling by duplication

        X_tr, Y_tr = oversample.fit_resample(X_tr, Y_tr)
        X_tr = np.array(X_tr)
        
        # Training associative memory
        assoc_mem = np.zeros((2, dim))
        for i in range(len(Y_tr)):
            assoc_mem[Y_tr[i]] += X_tr[i]
            
        assoc_mem = retrain(assoc_mem, X_tr, Y_tr, epochs=epochs, dim=dim, threshold=threshold)
        
        Y_pred, Y_score = inference(assoc_mem, X_te, Y_te, dim=10000)
        
        
        # Metrics
        auroc = sklearn.metrics.roc_auc_score(Y_te, Y_score)
        if auroc > max_auroc:
            max_assoc_mem = assoc_mem
            max_auroc = auroc
        
        metrics_dict["accuracy_list"].append(sklearn.metrics.accuracy_score(Y_te, Y_pred))
        metrics_dict["auroc_list"].append(sklearn.metrics.roc_auc_score(Y_te, Y_score))
        metrics_dict["bacc_list"].append(sklearn.metrics.balanced_accuracy_score(Y_te, Y_pred))
        metrics_dict["f1_list"].append(sklearn.metrics.f1_score(Y_te, Y_pred))
        metrics_dict["precision_list"].append(sklearn.metrics.precision_score(Y_te, Y_pred))
        metrics_dict["recall_list"].append(sklearn.metrics.recall_score(Y_te, Y_pred))
        metrics_dict["confusion_matrices"].append(sklearn.metrics.confusion_matrix(Y_te, Y_pred))
        metrics_dict["random_states"].append(random_state)
        
    print()
    print("Stats corresponding to Maximum accuracy are: ")

    max_auroc = max(metrics_dict["auroc_list"])
    max_auroc_idx = metrics_dict["auroc_list"].index(max_auroc)
    print("Accuracy: ", metrics_dict["accuracy_list"][max_auroc_idx])
    print("Auroc: ", metrics_dict["auroc_list"][max_auroc_idx])
    print("Bacc: ", metrics_dict["bacc_list"][max_auroc_idx])
    print("F1: ", metrics_dict["f1_list"][max_auroc_idx])
    print("Precision: ", metrics_dict["precision_list"][max_auroc_idx])
    print("Recall: ", metrics_dict["recall_list"][max_auroc_idx])
    print("Confusion Matrix: ", metrics_dict["confusion_matrices"][max_auroc_idx])
    print("Random State: ", metrics_dict["random_states"][max_auroc_idx])

    print()

    print("Stats corresponding to Minimum accuracy are: ")

    min_auroc = min(metrics_dict["auroc_list"])
    min_auroc_idx = metrics_dict["auroc_list"].index(min_auroc)
    print("Accuracy: ", metrics_dict["accuracy_list"][min_auroc_idx])
    print("Auroc: ", metrics_dict["auroc_list"][min_auroc_idx])
    print("Bacc: ", metrics_dict["bacc_list"][min_auroc_idx])
    print("F1: ", metrics_dict["f1_list"][min_auroc_idx])
    print("Precision: ", metrics_dict["precision_list"][min_auroc_idx])
    print("Recall: ", metrics_dict["recall_list"][min_auroc_idx])
    print("Confusion Matrix: ", metrics_dict["confusion_matrices"][min_auroc_idx])
    print("Random State: ", metrics_dict["random_states"][min_auroc_idx])

    print()

    print(f"Average Stats for {iterations} iterations")
    print("Accuracy: ", sum(metrics_dict["accuracy_list"])/iterations)
    print("Auroc: ", sum(metrics_dict["auroc_list"])/iterations)
    print("Bacc: ", sum(metrics_dict["bacc_list"])/iterations)
    print("F1: ", sum(metrics_dict["f1_list"])/iterations)
    print("Precision: ", sum(metrics_dict["precision_list"])/iterations)
    print("Recall: ", sum(metrics_dict["recall_list"])/iterations)


    dataset_file_suffix = dataset_file.split("/")[-1].split(".")[0]
    file_suffix = f"{dataset_file_suffix}_data_{target}_tar_{dim}_dim_{gramsize}_gm_{encoding_scheme}_{split_type}_{version}.p"

    print()
    print("Saving performance metrics dictionary and best performing model...")

    with open(f'./outputs/metrics_dict_{file_suffix}', 'wb') as f:
        pickle.dump(metrics_dict, f)


    with open(f'./models/model_{file_suffix}', 'wb') as f:
        pickle.dump(assoc_mem, f)
        
    print("Saving completed ...")