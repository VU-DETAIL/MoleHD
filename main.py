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



dataset = pd.read_csv('data/clintox.csv', sep=',', header=0)

X = dataset.smiles.to_numpy()
# Two tasks of Clintox
Y = dataset.CT_TOX.to_numpy()
# Y = dataset.FDA_APPROVED.to_numpy()

# HDC Hyperparameters
num_tokens = 1500
dim = 10000
max_pos = 256
gramsize = 3
epochs = 150

# Load pretrained SMILES-PE
spe_vob= codecs.open('data/SPE_ChEMBL.txt')
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

item_mem = np.random.choice((-1, 1), (num_tokens+1, dim))
pos_mem = np.random.choice((-1, 1), (max_pos, dim))

data_HV = []
for sample_ in data_tokenized:
    sample_ = sample_[:max_pos]
    sample_ = [item_mem[i] for i in sample_]
    sample_hv = np.ones_like(sample_[0])
    # sample_hv = sum(sample_) #plain addition
    for token_idx in range(len(sample_)):
    # for token_idx in range(len(sample_) - gramsize):
        # sample_hv = sample_hv + np.multiply(sample_[token_idx], sample_[token_idx+1])  # Bi-gram
        sample_hv = sample_hv + np.roll(sample_[token_idx], token_idx) #Uni-gram
        # sample_hv = sample_hv + np.multiply(sample_[token_idx], np.multiply(sample_[token_idx+1], sample_[token_idx+2])) # Tri-gram w/o roll
        # sample_hv = sample_hv + np.multiply(pos_mem[token_idx], sample_[token_idx]) # Record-based
    
    sample_hv[sample_hv>max_pos] = max_pos
    sample_hv[sample_hv<-max_pos] = -max_pos
    data_HV.append(sample_hv)

X_tr, X_te, Y_tr, Y_te= train_test_split(data_HV, Y, test_size=0.2) # , stratify=Y # Toggle straify and random split, can also use 5-fold. 
_, X_te, _, Y_te= train_test_split(X_te, Y_te, test_size=0.5) # 8/1/1 split

oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority') #Oversampling by duplication

X_tr, Y_tr = oversample.fit_resample(X_tr, Y_tr)
X_tr = np.array(X_tr)

# Training associative memory
assoc_mem = np.zeros((2, dim))
for i in range(len(Y_tr)):
    assoc_mem[Y_tr[i]] += X_tr[i]

# Retraining
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

# Inference
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

print(acc, bacc, f1, precision, recall, roc_auc)
print(sklearn.metrics.confusion_matrix(Y_te, Y_pred))

# Output figures
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve of MoleHD')
# plt.legend(loc="lower right")
# plt.show()
