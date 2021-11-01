import deepchem as dc
from deepchem.splits.splitters import ScaffoldSplitter
import codecs
from SmilesPE.tokenizer import *
import numpy as np
import random
from scipy.spatial.distance import cdist
from collections import Counter


def clean_dataset(X, Y):
    """Cleans the molecules from the dataset using deepchem.

    This function takes the full dataset and uses deepchem to filter out bad 
    molecules. Bad molecules are those that deepchem thinks are not valid molecules
    because of various reasons such as atoms have more bonding than permitted. One of 
    such molecule is [NH4][Pt]([NH4])(Cl)Cl, for which deepchem throws error saying 
    "... Explicit valence for atom # 0 N, 5, is greater than permitted"

    Args:
        param X (list): the list of all molecules in the dataset
        param Y (list): the label of all molecules in the dataset
    
    Returns:
        X_Clean (list): cleaned list of molecules
        Y_clean (list): labels for cleaned molecules
        X_bad (list): bad list of molecules
        Y_bad (list): labels for bad molecules
    """
    
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


def data_tokenize_smiles_pretrained(X, num_tokens=1500):
    """Tokenizes the entire dataset using smiles pretained tokenization

    This function takes complete dataset X and number of maximum tokens and tokenizes
    the dataset into list numerical numbers representing molecules in the dataset. First, 
    a lookup dictionary is created using the entire dataset and smiles pretrained tokenization model.
    At max, num_tokens are included in the dictionary based on their count. Later, while tokenizating 
    the molecules in the dataset, molecule is split into parts into using smiles pretrained model, 
    the lookup dictionary is used to assign number to each of these parts from a molecule. 

    Args:
        param X (list): the list of all molecules in the dataset
        num_tokens (int): maximum number of tokens to allow in the lookup dictionary

    Returns:
         data_tokenized (list): Returns list of list, with each list representing tokenized molecule
    """


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
        #print(sample_token)
    return data_tokenized


def data_tokenize_atomwise(X, num_tokens=1500):
    """Tokenizes the entire dataset using atomwise tokenization method

    This function takes complete dataset X and number of maximum tokens and tokenizes
    the dataset into list numerical numbers representing molecules in the dataset. First, 
    a lookup dictionary is created using the entire dataset and smiles pretrained tokenization model 
    used to break the molecule into individual atom parts. At max, num_tokens are included in the
    dictionary based on their count. Later, while tokenizating the molecules in the dataset, molecule is 
    split into atoms using the same smiles pretrained model, the lookup dictionary is used to 
    assign number to each of these parts from a molecule. 

    Args:
        param X (list): the list of all molecules in the dataset
        num_tokens (int): maximum number of tokens to allow in the lookup dictionary

    Returns:
         data_tokenized (list): Returns list of list, with each list representing tokenized molecule
    """
    
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
    """Tokenizes the entire dataset using characterwise tokenization method

    This function takes complete dataset X and number of maximum tokens and tokenizes
    the dataset into list of numerical numbers representing molecules in the dataset. First, 
    the entire dataset is converted into characters and a count dictionary is generated for
    each unique character in the dataset. At max, 1500 unique characters with maximum frequency 
    is included in the dictionary. Later, while tokenizating the molecules in the dataset, molecule is 
    split into characters, the lookup dictionary is used to assign number to each of these parts from a molecule. 

    Args:
        param X (list): the list of all molecules in the dataset
        num_tokens (int): maximum number of tokens to allow in the lookup dictionary

    Returns:
         data_tokenized (list): Returns list of list, with each list representing tokenized molecule
    """
    
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


def create_data_HV(data_tokenized, gramsize=1, num_tokens=1500, dim=10000, max_pos=256, random_state=0):
    """Creates hypervector from the tokenized molecules

    This function takes the tokenized dataset and gramsize for n-gram encoding method, and creates
    a high dimensional vector representation of the molecule. A lookup high-dimensional memory called
    item_mem is created with size (num_tokens+1, dim) and value -1/1. Dim is the dimension of the hypervector
    which is usually 10000. Using the numerical value in the token representation of the molecule, 
    a hypervector is taken from the item memory. Eventually, the molecule will be represented as a list
    of hypervector, which is then aggregated using n-gram encoding scheme.

    Args:
        data_tokenized (list): list of list, with each list representing tokenized molecule
        gramsize (int): number for n-gram encoding
        num_tokens: maximum number of tokens 
        dim: dimension of hypervector
        max_pos: maximum position value
        random_state: random state for generating hypervector
    
    Returns:
        data_HV (list): list of hypervector, with each hypervector representing corresponding molecule in X
    """
    
    random.seed(random_state)
    item_mem = np.random.choice((-1, 1), (num_tokens+1, dim))

    data_HV = []
    for sample_ in data_tokenized:
        sample_ = sample_[:max_pos]
        sample_ = [item_mem[i] for i in sample_]
        sample_hv = np.ones_like(sample_[0])
        
        for token_idx in range(len(sample_) - gramsize + 1):
            if gramsize == 1:
                #sample_hv = sample_hv + np.roll(sample_[token_idx], token_idx)\
                sample_hv = sample_hv + sample_[token_idx]
            elif gramsize == 2:
                sample_hv = sample_hv + np.multiply(sample_[token_idx], sample_[token_idx+1])
            elif gramsize == 3:
                sample_hv = sample_hv + np.multiply(sample_[token_idx], np.multiply(sample_[token_idx+1], sample_[token_idx+2]))
            elif gramsize == 4:
                sample_hv = sample_hv + np.multiply(np.multiply(sample_[token_idx], np.multiply(sample_[token_idx+1], sample_[token_idx+2])), sample_[token_idx+3])

        sample_hv[sample_hv>max_pos] = max_pos
        sample_hv[sample_hv<-max_pos] = -max_pos
        data_HV.append(sample_hv)
        
    return data_HV

def train_test_split_scaffold(X, Y, data_HV, test_size=0.20, random_state=10):
    """Splits the dataset into training and testing set based scaffold split method

    This function takes the entire dataset and splits the dataset into training and 
    testing dataset based on scaffold split method. It also takes the data_HV, which is
    the list of hypervector representing each molecule in the dataset and splits it into
    training and testing hypervectors. In order to generate the indices for training and
    testing dataset, we first need to convert the entire original molecule dataset into
    numpy dataset and supply it to split function in deepchem's scaffold splitter. 

    Args:
        param X (list): the list of all molecules in the dataset
        param Y (list): the label of all molecules in the dataset
        data_HV (list): list of hypervector, with each hypervector representing corresponding molecule in X
        test_size (float): train test split ratio
        random_state: random state for generating hypervector
    
    Returns: 
        X_tr (list): training dataset with hypervector representing molecules
        X_te (list): testing dataset with hypervector representing molecules
        Y_tr (list): the label of all molecules in the training dataset
        Y_te (list): the label of all molecules in the testing dataset
    """
    
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


def retrain(assoc_mem, X_tr, Y_tr, epochs=150, dim=10000, threshold=1024):
    """retrains the associative memory for a given number of epochs

    This function takes the trained associative memory and retrains the memory
    for epochs. On each epoch, the function tries to fix the wrong label by subtracting
    the misclassified hypervector from the wrong label and adding it to the correct label 
    in the associative memory.

    Args:
        assoc_mem (list): An associative memory of size (n, dim), where n is the number of labels
        X_tr (list): training dataset with hypervector representing molecules
        Y_tr (list): the label of all molecules in the training dataset
        epochs (int): number of times to run the loop in order to fix errors
        dim (int): dimension of hypervector
        threshold (int): value used to chip the maximum and minimum value in associative memory
    
    Returns: 
        assoc_mem (list): retrained associative memory
    """
    
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
    """generates prediction and score on testing dataset

    Args:
        assoc_mem (list): An associative memory of size (n, dim), where n is the number of labels
        X_te (list): testing dataset with hypervector representing molecules
        Y_te (list): the label of all molecules in the testing dataset
        dim (int): dimension of hypervector
    
    Returns:
        Y_pred (list): predictions on testing set
        Y_score (list): prediction scores on testing set
    """
    
    Y_pred = []
    Y_score = []
    for i in range(len(Y_te)):
        dist = cdist(X_te[i].reshape(1, dim), assoc_mem, metric='cosine')
        dist = dist[0]
        if dist[0] > dist[1]:
            pred = 1
        else:
            pred = 0
        Y_pred.append(pred)
        Y_score.append((dist[0] - dist[1] + 2)/ 4)
        
    return (Y_pred, Y_score)
