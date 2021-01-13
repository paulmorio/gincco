"""
Script for running experiments with SVMs and FC MLP on Metabric dataset
"""

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
from data.metabric_loader import load_metabric
from methods.svm import rbf_svm_classify 

# #########################################################
# # Single SVM example with small 1k genes set for testing
# #########################################################
# data_path = "data/1k_full.csv"
# task = "PAM50"
# preprocessing = "scale"

# # Grab the metabric data
# x, y, genes, index_to_genesymbol, genesymbol_to_index = load_metabric(data_path, task, preprocessing)

# # Set up the experiment data
# kf = StratifiedKFold(n_splits=5)
# acc_results = []
# counter = 0
# for train_ids, test_ids in kf.split(x, y):
#     print("## Starting experiments on fold {}".format(counter))
#     x_train = x[train_ids]
#     x_test = x[test_ids]
#     y_train = y[train_ids]
#     y_test = y[test_ids]
#     scores = rbf_svm_classify(x_train, x_test, y_train, y_test)
#     acc_results.append(scores[0])
#     counter += 1
# print("## 5 fold accuracy on target: %s mean: %f stddev: %f" %  (task, np.mean(acc_results), np.std(acc_results)))



# ######################
# # Full Cour with SVM
# ######################
# data_path = "data/MBdata_all.csv"
# preprocessing = "scale"
# base_case_scores = {}
# for target in ["DR","ER","PAM50","IC10"]:
# # for target in ["PR"]:
#     x, y, genes, index_to_genesymbol, genesymbol_to_index = load_metabric(data_path, target, preprocessing)
#     kf = StratifiedKFold(n_splits=5)
#     acc_results = []
#     counter = 0
#     for train_ids, test_ids in kf.split(x, y):
#         print("## Starting experiments on fold {}".format(counter))
#         x_train = x[train_ids]
#         x_test = x[test_ids]
#         y_train = y[train_ids]
#         y_test = y[test_ids]
#         scores = rbf_svm_classify(x_train, x_test, y_train, y_test)
#         acc_results.append(scores[0])
#         counter += 1
#     print("## 5 fold accuracy on target: %s mean: %f stddev: %f" %  (target, np.mean(acc_results), np.std(acc_results)))
#     base_case_scores[target] = (np.mean(acc_results), np.std(acc_results))

# print("########\n##.. Done full course SVM for base case on raw gene expression \n#########")
# for target in base_case_scores.keys():
#     print("## 5 fold accuracy on target: %s mean: %f stddev: %f" %  (target, base_case_scores[target][0], base_case_scores[target][1]))



######################
# Full Cour with FC-MLP
######################
# Data settings
data_path = "data/MBdata_all.csv"
preprocessing = "scale"

# Device settings
dtype = torch.float
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Model Definition
class SimpleNet(nn.Module):
    def __init__(self, Din, H, Dout):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(Din, H)
        self.linear2 = nn.Linear(H, Dout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h_relu = torch.relu(self.linear1(x))
        y_pred_raw = torch.relu(self.linear2(h_relu))
        y_pred = self.softmax(y_pred_raw)
        return y_pred

# Experiments
base_case_scores = {}
# for target in ["DR","ER","PAM50","IC10"]:
for target in ["PR"]:
    # load metabric data and metas
    x, y, genes, index_to_genesymbol, genesymbol_to_index = load_metabric(data_path, target, preprocessing)
    
    # Data as Torch tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y)

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

        N, Din = x_train.shape
        H = 1600
        Dout =  len(set(y_train.tolist()))

        yints = y_train.squeeze().data.numpy()
        class_weights = class_weight.compute_class_weight('balanced', np.unique(yints), yints)
        cweights = torch.tensor(list(class_weights), dtype=torch.float)
        cweights = cweights.to(device)

        model = SimpleNet(Din, H, Dout).to(device)
        # criterion = nn.CrossEntropyLoss(reduction='none')
        criterion = nn.CrossEntropyLoss(weight = cweights)
        lr = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.1, verbose=True)
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

print("########\n##.. Done full course SVM for base case on raw gene expression \n#########")
for target in base_case_scores.keys():
    print("## 5 fold accuracy on target: %s mean: %f stddev: %f" %  (target, base_case_scores[target][0], base_case_scores[target][1]))

