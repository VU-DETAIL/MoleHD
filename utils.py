import deepchem as dc
from deepchem.splits.splitters import ScaffoldSplitter
import codecs
from SmilesPE.tokenizer import *
import numpy as np
import random
from scipy.spatial.distance import cdist
from collections import Counter


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
    
    return (X_clean, Y_clean, X_bad, Y_bad)


def train_test_split_scaffold(X, Y, data_HV, test_size=0.20, random_state=10):
    
    frac_valid = test_size/2
    frac_test = test_size/2
    frac_train = 1 - test_size
    
    splitter = dc.splits.ScaffoldSplitter()
    dataset = dc.data.NumpyDataset(X=np.array(X), ids=X)
    train_idxs, valid_idxs, test_idxs = splitter.split(dataset, 
                                                       frac_train=frac_train, 
                                                       frac_valid=frac_valid, 
                                                       frac_test=frac_test)
    
        
    X_tr = []
    Y_tr = []
    X_te = []
    Y_te = []

    for idx in train_idxs:
        X_tr.append(data_HV[idx])
        Y_tr.append(Y[idx])

    for idx in valid_idxs:
        X_te.append(data_HV[idx])
        Y_te.append(Y[idx])

    for idx in test_idxs:
        X_te.append(data_HV[idx])
        Y_te.append(Y[idx])
    
    return (X_tr, X_te, Y_tr, Y_te)


def data_tokenize_smiles_pretrained(X, num_tokens=1500):
    
    # Load pretrained SMILES-PE
    spe_vob = codecs.open('./data/SPE_ChEMBL.txt')
    spe = SPE_Tokenizer(spe_vob)

    # Build dictionary using all the data
    scopus = [] 

    for sample in X:
        sample_token = spe.tokenize(sample).split(' ')
        scopus = scopus + sample_token
        
    dict = Counter(scopus).most_common(num_tokens)
    dict = {x[0]: i for i, x in enumerate(dict)}
        
    data_tokenized = []
    for sample in X:
        sample_token = spe.tokenize(sample).split(' ')
        sample_token = [dict[x] if x in dict else num_tokens for x in sample_token]
        data_tokenized.append(sample_token)
    
    return data_tokenized


def data_tokenize_atomwise(X, num_tokens=1500):
    
    # Build dictionary using all the data
    scopus = []

    for smile in X:
        toks = atomwise_tokenizer(smile)
        scopus += toks

    if len(set(scopus)) < num_tokens:
        num_tokens = len(set(scopus))

    dict = Counter(scopus).most_common(num_tokens)
    dict = {x[0]: i for i, x in enumerate(dict)}

    data_tokenized = []
    for sample in X:
        sample_token = atomwise_tokenizer(sample)
        sample_token = [dict[x] if x in dict else num_tokens for x in sample_token]
        data_tokenized.append(sample_token)
        
    return data_tokenized


def data_tokenize_characterwise(X, num_tokens=1500):
    
    # Build dictionary using all the data
    scopus = []

    for smile in X:
        scopus += list(smile)

    if len(set(scopus)) < num_tokens:
        num_tokens = len(set(scopus))

    dict = Counter(scopus).most_common(num_tokens)
    dict = {x[0]: i for i, x in enumerate(dict)}

    data_tokenized = []
    for sample in X:
        sample_token = list(sample)
        sample_token = [dict[x] if x in dict else num_tokens for x in sample_token]
        data_tokenized.append(sample_token)
       
    return data_tokenized


def create_associative_memory(data_tokenized, gramsize=1, num_tokens=1500, dim=10000, max_pos=256, random_state=10):
    
    random.seed(random_state)
    item_mem = np.random.choice((-1, 1), (num_tokens+1, dim))

    data_HV = []
    for sample_ in data_tokenized:
        sample_ = sample_[:max_pos]
        sample_ = [item_mem[i] for i in sample_]
        sample_hv = np.ones_like(sample_[0])
        
        for token_idx in range(len(sample_) - gramsize + 1):

            if gramsize == 1:
                sample_hv = sample_hv + np.roll(sample_[token_idx], token_idx)
            elif gramsize == 2:
                sample_hv = sample_hv + np.multiply(sample_[token_idx], sample_[token_idx+1])
            elif gramsize == 3:
                sample_hv = sample_hv + np.multiply(sample_[token_idx], np.multiply(sample_[token_idx+1], sample_[token_idx+2]))

        sample_hv[sample_hv>max_pos] = max_pos
        sample_hv[sample_hv<-max_pos] = -max_pos
        data_HV.append(sample_hv)
        
    return data_HV


def retrain(assoc_mem, X_tr, Y_tr, epochs=150, dim=10000, threshold=1024):
    
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
    assoc_mem[assoc_mem < -threshold] = -threshold
    assoc_mem[assoc_mem > threshold] = threshold
    
    return assoc_mem


def inference(assoc_mem, X_te, Y_te, dim=10000):
    
    # Inference
    Y_pred = []
    Y_score = []
    for i in range(len(Y_te)):
        dist = cdist(X_te[i].reshape(1, dim), assoc_mem, metric='cosine')
        dist = dist[0]
        if dist[0] < dist[1]:
            pred = 0
        else:
            pred = 1
        Y_pred.append(pred)
        Y_score.append((dist[0] - dist[1] + 2)/ 4)
        
    return (Y_pred, Y_score)
    
    

