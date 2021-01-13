"""Module investigating Gincco architecture if we used random clustering and sparsity 
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
from CustomizedLinear import CustomizedLinear
from data.metabric_loader import load_tcga, load_metabric
from methods.svm import rbf_svm_classify 
from data.utils import get_generic_string_overlap_network

#########################################################
# Reproducibility Settings 
#########################################################
torch.manual_seed(123)
np.random.seed(123)
split_seed = 123

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

for _ in tqdm(range(25)):
    # Defining the random computational graph to evaluate
    data_path = "data/tcga_hncs.csv"
    preprocessing = "scale"

    # Define model based on mask
    class CustomSimpleNet(nn.Module):
        def __init__(self, Din, H, Dout, mask):
            super(CustomSimpleNet, self).__init__()
            self.customlinear = CustomizedLinear(mask, bias=None)
            self.linearh = nn.Linear(H, H)
            self.linear = nn.Linear(H, Dout)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            h_relu = torch.relu(self.customlinear(x))
            # y_pred_raw = torch.relu(self.linearh(h_relu))
            y_pred_raw = torch.relu(self.linear(h_relu))
            y_pred = self.softmax(y_pred_raw)
            return y_pred

    # Generate the random mask constant throughout the experiments
    num_random_clusters = np.random.randint(1, 60)
    x, y, genes, index_to_genesymbol, genesymbol_to_index = load_tcga(data_path, "tumor_grade", "scale")
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
    for target in ["tumor_grade", "X2yr.RF.Surv."]:
        # Grab the tcga data
        x, y, genes, index_to_genesymbol, genesymbol_to_index = load_tcga(data_path, target, preprocessing)

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

    metrics_to_write = []
    metrics_to_write.append(num_random_clusters)
    metrics_to_write.append(num_random_connections)
    print("########\n##.. Done full course RandomNN for base case on raw gene expression \n#########")
    for target in sorted(list(base_case_scores.keys())):
        print("## 5 fold accuracy on target: %s mean: %f stddev: %f" %  (target, base_case_scores[target][0], base_case_scores[target][1]))
        metrics_to_write.append(base_case_scores[target][0])
        metrics_to_write.append(base_case_scores[target][1])
        metrics_to_write.append(base_case_scores[target][2])
        metrics_to_write.append(base_case_scores[target][3])
        metrics_to_write.append(base_case_scores[target][4])
        metrics_to_write.append(base_case_scores[target][5])
        metrics_to_write.append(base_case_scores[target][6])
        metrics_to_write.append(base_case_scores[target][7])
        metrics_to_write.append(base_case_scores[target][8])
        metrics_to_write.append(base_case_scores[target][9])

    with open("tcga_random_pcomplex.csv", "a") as fh:
        fh.write(",".join([str(s) for s in metrics_to_write]) + "\n")