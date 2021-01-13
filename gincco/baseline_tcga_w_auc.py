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
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, roc_auc_score
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
from data.metabric_loader import load_tcga
from methods.svm import rbf_svm_classify, rbf_svm_classify_roc, rbf_svm_classify_roc_binary

#########################################################
# Reproducibility Settings 
#########################################################
torch.manual_seed(123)
np.random.seed(123)
split_seed = 123



######################
# Full Cour with SVM
######################
data_path = "data/tcga_hncs.csv"
preprocessing = "scale"
base_case_scores = {}
for target in ["tumor_grade", "X2yr.RF.Surv."]:    
    x, y, genes, index_to_genesymbol, genesymbol_to_index = load_tcga(data_path, target, preprocessing)
    kf = StratifiedKFold(n_splits=5, random_state=split_seed)
    acc_results = []
    precision_results = []
    recall_results = []
    f_score_results = []
    roc_auc_scores = []
    counter = 0
    for train_ids, test_ids in kf.split(x, y):
        print("## Starting experiments on fold {}".format(counter))
        x_train = x[train_ids]
        x_test = x[test_ids]
        y_train = y[train_ids]
        y_test = y[test_ids]
        if target == "X2yr.RF.Surv.":
            scores = rbf_svm_classify_roc_binary(x_train, x_test, y_train, y_test)
        else:
            scores = rbf_svm_classify_roc(x_train, x_test, y_train, y_test)
        acc_results.append(scores[0])
        precision_results.append(scores[1])
        recall_results.append(scores[2])
        f_score_results.append(scores[3])
        roc_auc_scores.append(scores[4])
        counter += 1
    # print("## 5 fold accuracy on target: %s mean: %f stddev: %f" %  (target, np.mean(acc_results), np.std(acc_results)))
    base_case_scores[target] = (np.mean(acc_results), np.std(acc_results), 
        np.mean(precision_results), np.std(precision_results),
        np.mean(recall_results), np.std(recall_results),
        np.mean(f_score_results), np.std(f_score_results),
        np.mean(roc_auc_scores), np.std(roc_auc_scores))

print("########\n##.. Done full course SVM for base case on raw gene expression \n#########")
for target in base_case_scores.keys():
    # print("## 5 fold accuracy on target: %s mean: %f stddev: %f" %  (target, base_case_scores[target][0], base_case_scores[target][1]))
    print("""## 5 fold mean accuracy on target: %s mean: %f stddev: %f
        5 fold mean precision: %f, stddev: %f
        5 fold mean recall: %f, stddev: %f
        5 fold mean fscore: %f, stddev: %f
        5 fold mean roc-auc: %f, stddev: %f""" %  (target, base_case_scores[target][0], base_case_scores[target][1],
                                                base_case_scores[target][2], base_case_scores[target][3],
                                                base_case_scores[target][4], base_case_scores[target][5],
                                                base_case_scores[target][6], base_case_scores[target][7],
                                                base_case_scores[target][8], base_case_scores[target][9]))


######################
# Full Cour with FC-MLP
######################
# Data settings
data_path = "data/tcga_hncs.csv"
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
for target in ["tumor_grade", "X2yr.RF.Surv."]: 

    # load metabric data and metas
    x, y, genes, index_to_genesymbol, genesymbol_to_index = load_tcga(data_path, target, preprocessing)
    
    # Data as Torch tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y)

    # Set up experiment logs
    kf = StratifiedKFold(n_splits=5, random_state=split_seed)
    acc_results = []
    precision_results = []
    recall_results = []
    f_score_results = []
    roc_auc_scores = []
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
            accuracy = accuracy_score(y_test.data, indices)
            precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test.data, indices)
            if target == "X2yr.RF.Surv.":
                roc_auc = roc_auc_score(y_test.data, indices)
            else:
                roc_auc = roc_auc_score(y_test.data.numpy(), outputs.numpy(), multi_class="ovr")

        acc_results.append(accuracy)
        precision_results.append(precision)
        recall_results.append(recall)
        f_score_results.append(fbeta_score)
        roc_auc_scores.append(roc_auc)
        counter += 1

    print("## 5 fold accuracy on target: %s mean: %f stddev: %f" %  (target, np.mean(acc_results), np.std(acc_results)))
    base_case_scores[target] = (np.mean(acc_results), np.std(acc_results), 
        np.mean(precision_results), np.std(precision_results),
        np.mean(recall_results), np.std(recall_results),
        np.mean(f_score_results), np.std(f_score_results),
        np.mean(roc_auc_scores), np.std(roc_auc_scores))

print("########\n##.. Done full course FC MLP for base case on raw gene expression \n#########")
for target in base_case_scores.keys():
    # print("## 5 fold accuracy on target: %s mean: %f stddev: %f" %  (target, base_case_scores[target][0], base_case_scores[target][1]))
    print("""## 5 fold mean accuracy on target: %s mean: %f stddev: %f
        5 fold mean precision: %f, stddev: %f
        5 fold mean recall: %f, stddev: %f
        5 fold mean fscore: %f, stddev: %f
        5 fold mean roc-auc: %f, stddev: %f""" %  (target, base_case_scores[target][0], base_case_scores[target][1],
                                                base_case_scores[target][2], base_case_scores[target][3],
                                                base_case_scores[target][4], base_case_scores[target][5],
                                                base_case_scores[target][6], base_case_scores[target][7],
                                                base_case_scores[target][8], base_case_scores[target][9] ) )