"""Module investigating gincco architecture if we used random clustering and sparsity 
assignments
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, normalize, scale, StandardScaler, MinMaxScaler
from collections import defaultdict
from tqdm import tqdm

# For model
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN, AgglomerativeClustering

# Pytorch utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import class_weight

# Internal
from CustomizedLinear import CustomizedLinear
from data.metabric_loader import load_tcga, load_metabric
from methods.svm import rbf_svm_classify 
from data.utils import get_generic_string_overlap_network

######################
# Data and General Experiment Parameters
######################
data_path = "data/MBdata_all.csv"
target = "Grade"
preprocessing = "scale"

######################
# Utilities
######################
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

#################################
## Device settings for Pytorch ##
#################################
dtype = torch.float
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Define model based on mask
class CustomSimpleNet(nn.Module):
    def __init__(self, Din, H, Dout, mask):
        super(CustomSimpleNet, self).__init__()
        self.customlinear = CustomizedLinear(mask, bias=None)
        self.linearh = nn.Linear(H, H)
        self.linear = nn.Linear(H, Dout)
        self.softmax = nn.Softmax(dim=1)

        # torch.nn.init.xavier_uniform(customlinear.weight)
        # torch.nn.init.xavier_uniform(linear.weight)

    def forward(self, x):
        h_relu = torch.relu(self.customlinear(x))
        # y_pred_raw = torch.relu(self.linearh(h_relu))
        y_pred_raw = torch.relu(self.linear(h_relu))
        y_pred = self.softmax(y_pred_raw)
        return y_pred

# Generate the random mask constant throughout the experiments
num_random_clusters = np.random.randint(6000)
x, y, genes, index_to_genesymbol, genesymbol_to_index = load_metabric(data_path, target, preprocessing)
adj = np.zeros((len(list(genesymbol_to_index.keys())), num_random_clusters))
num_random_connections = np.random.randint(np.prod(adj.shape))

print("Generating Random Computation Graph Structure")
counter = num_random_connections
m,n = adj.shape
for i in tqdm(range(counter)):
    not_assigned = True
    while not_assigned:
        rm = np.random.randint(m)
        rn = np.random.randint(n)
        if adj[rm][rn] == 0:
            adj[rm][rn] = 1
            not_assigned = False
        else:
            continue

mask = torch.from_numpy(adj).int()
print("Computation graph contains %f random clusters with %f randomly assigned gene to module assignments" % (num_random_clusters, num_random_connections))


# Experiments
base_case_scores = {}
# for target in ["DR","ER","PAM50","IC10", "Grade", "PR"]:
for target in ["Grade"]:
    # Grab the metabric data
    x, y, genes, index_to_genesymbol, genesymbol_to_index = load_metabric(data_path, target, preprocessing)

    # Data as Torch tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y)

    Dim_INPUT  = mask.shape[0]
    Dim_HIDDEN = mask.shape[1]
    Dim_OUTPUT = len(set(y.tolist()))

    print(mask.shape)
    print(Dim_INPUT)
    print(Dim_HIDDEN)
    print(Dim_OUTPUT)

    # Set up experiment logs
    kf = StratifiedKFold(n_splits=5)
    acc_results = []
    counter = 0
    for train_ids, test_ids in kf.split(x, y):
        print("## Starting experiments on fold {}".format(counter))
        x_train = x[train_ids]
        x_test = x[test_ids]
        y_train = y[train_ids]
        y_test = y[test_ids]

        epochs = 250
        mini_batch_size = 32
        
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=False)

        yints = y_train.squeeze().data.numpy()
        class_weights = class_weight.compute_class_weight('balanced', np.unique(yints), yints)
        cweights = torch.tensor(list(class_weights), dtype=torch.float)
        cweights = cweights.to(device)

        model = CustomSimpleNet(Dim_INPUT, Dim_HIDDEN, Dim_OUTPUT, mask).to(device)
        model.apply(init_weights)

        # criterion = nn.CrossEntropyLoss(reduction='none')
        criterion = nn.CrossEntropyLoss(weight = cweights)
        lr = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, verbose=True)
        for epoch in range(epochs):
            losses = []

            # # for xb, yb in train_loader:
            # optimizer.zero_grad()

            # # Forward pass
            # x_train = x_train.to(device)
            # y_train = y_train.to(device)
            # y_pred = model(x_train)

            # # Compute and print loss
            # loss = criterion(y_pred, y_train)
            # losses.append(loss.item())

            # # Zero gradients, perform the backward pass and update the weights.
            # loss.backward()
            # optimizer.step()


            for xb, yb in train_loader:
                optimizer.zero_grad()
                xb = xb.to(device)
                yb = yb.to(device)
                y_pred = model(xb)

                # Compute and print loss
                loss = criterion(y_pred, yb)
                losses.append(loss.item())


                # Zero gradients, perform the backward pass and update the weights.
                loss.backward()
                optimizer.step()

            mean_loss = sum(losses)/len(losses)
            # scheduler.step(mean_loss)

            if epoch % 50 == 0:
                print("Epoch {} at loss {}".format(epoch, mean_loss))

        with torch.no_grad():
            outputs = model(x_test.to(device)).cpu()
            values, indices = torch.max(outputs.data, 1)
            accuracy = accuracy_score(indices, y_test.data)

        acc_results.append(accuracy)
        counter += 1

    print("## 5 fold accuracy on target: %s mean: %f stddev: %f" %  (target, np.mean(acc_results), np.std(acc_results)))
    base_case_scores[target] = (np.mean(acc_results), np.std(acc_results))

print("########\n##.. Done full course NN for base case on raw gene expression \n#########")
for target in base_case_scores.keys():
    print("## 5 fold accuracy on target: %s mean: %f stddev: %f" %  (target, base_case_scores[target][0], base_case_scores[target][1]))

