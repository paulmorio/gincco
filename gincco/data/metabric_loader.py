"""Loads data specific for Metabric CSV data as pandas and numpy instances
with some metadata about the names of the genes involved
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, normalize, scale, StandardScaler, MinMaxScaler
from collections import defaultdict
from tqdm import tqdm


def load_metabric(csv_path, task, preprocessing):
    """Load metabric data as numpy ndarray matrices
    
    Parameters
    ----------
    csv_path : str
        Path to metabric csv file such as "MBdata_all.csv"
    task : str
        String denotation for the target like "PAM50"
    preprocessing : str
        String denotation for the preprocessing to perform on 
        the numeric gene expressions possible values are
        {"none", "minmax", "scale", "normalise"}

    Returns
    -------
    (x, y, gene_symbols, index_to_genesymbol, genesymbol_to_index) : tuple
        tuple containing x, y as numpy arrays. gene_symbols is list of 
        string names of genes, index_to_genesybol and vice versa are dictionaries
        connecting string genesymbols to their index on the x matrix

    """
    data_path = csv_path

    print("## Loading data into memory")
    df = pd.read_csv(data_path)
    if task == 'DR':
        df = df[df.DR != '?']
        y = df.pop('DR')

    elif task == 'ER':
        df = df[df.ER_Status != '?']
        y = df.pop('ER_Status')
        labels = {
            'pos': 0,
            'neg': 1
        }
        y = y.apply(lambda x: labels[x])

    elif task == 'IC10':
        df = df[df.iC10 != '?']
        y = df.pop('iC10')
        labels = {
            '4ER-': 4,
            '4ER+': 0
        }
        y = y.apply(lambda x: labels[x] if x in labels else int(x))

    elif task == 'PAM50':
        df = df[df.Pam50Subtype != '?']
        y = df.pop('Pam50Subtype')
        pam50_lables = {
            'Normal': 0,
            'LumA': 1,
            'LumB': 2,
            'Basal': 3,
            'Her2': 4
        }
        y = y.apply(lambda x: pam50_lables[x])

    elif task == "Grade":
        df = df[df["Grade"] != '?']
        y = df.pop('Grade')
        y = y.astype(str)
        grade_labels = {
            '1' : 0,
            '2' : 1,
            '3' : 2
        }
        y = y.apply(lambda x: grade_labels[x])

    elif task == "PR":
        df = df[df["PR_Expr"] != "?"]
        y = df.pop("PR_Expr")
        y = y.astype(str)
        pr_labels = {
            "-" : 0,
            "+" : 1
        }
        y = y.apply(lambda x: pr_labels[x])

    x = df.loc[:, df.columns.str.startswith('GE')]
    x = x.fillna(x.mean()) # x has 10 missing values for Gene expressions, since its so low we just use fillna mean

    # Metadata
    gene_symbols = [gs[3:] for gs in x.columns]
    index_to_genesymbol = dict(enumerate(gene_symbols))
    genesymbol_to_index = {v: k for k, v in index_to_genesymbol.items()}

    x = np.array(x, dtype='float32')
    y = y.values

    print("## Preprocessing data")
    if preprocessing == "normalise":        
        # normalise or scale the column values
        x = normalize(x, norm="l2", axis=1, copy=True)

    elif preprocessing == "scale":
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)

    elif preprocessing == "minmax":
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(x)
        x = scaler.transform(x)

    return (x, y, gene_symbols, index_to_genesymbol, genesymbol_to_index)


def load_tcga(csv_path, task, preprocessing):
    print("## Loading data into memory")
    df = pd.read_csv(csv_path)
    if task == 'tumor_grade':
        df = df[df.tumor_grade != 'GX']
        y = df.pop('tumor_grade')
        tumor_grade_labels = {
            'G1' : 0,
            'G2' : 1,
            'G3' : 2,
            'G4' : 3
        }
        y = y.apply(lambda x: tumor_grade_labels[x])

    elif task == 'X2yr.RF.Surv.':
        # Relapse free survival (2 years)
        df = df[df['X2yr.RF.Surv.'] != '?']
        y = df.pop('X2yr.RF.Surv.')

    elif task == 'ajcc_metastasis_clinical_cm':
        # clinically diagnosed metastasis
        df = df[df['ajcc_metastasis_clinical_cm'] != 'MX']
        y = df.pop('ajcc_metastasis_clinical_cm')
        metastasis_labels = {
            'M0' : 0,
            'M1' : 1,
        }
        y = y.apply(lambda x: metastasis_labels[x])

    elif task == 'class':
        # Pancan only class of cancer (BRCA etc)
        df = df[df['class'] != '?']
        y = df.pop('class')
        class_labels = {
            'PRAD' : 0,
            'LUAD' : 1,
            'BRCA' : 2,
            'KIRC' : 3,
            'COAD' : 4
        }
        y = y.apply(lambda x: class_labels[x])

    x = df.loc[:, df.columns.str.contains('[a-zA-Z]+.*\|\d+')]
    x = x.fillna(x.mean()) # x has 10 missing values for Gene expressions, since its so low we just use fillna mean

    # Metadata
    gene_symbols = [gs[:gs.index('|')] for gs in x.columns]
    x.columns = gene_symbols
    x = x.loc[:,~x.columns.duplicated()]
    gene_symbols = x.columns

    index_to_genesymbol = dict(enumerate(gene_symbols))
    # genesymbol_to_index = {v: k for k, v in index_to_genesymbol.items()}
    genesymbol_to_index = {}
    for key in index_to_genesymbol.keys():
        if index_to_genesymbol[key] in genesymbol_to_index:
            print(index_to_genesymbol[key])
        else:
            genesymbol_to_index[index_to_genesymbol[key]] = key

    x = np.array(x, dtype='float32')
    y = y.values

    print("## Preprocessing data")
    if preprocessing == "normalise":        
        # normalise or scale the column values
        x = normalize(x, norm="l2", axis=1, copy=True)

    elif preprocessing == "scale":
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)

    elif preprocessing == "minmax":
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(x)
        x = scaler.transform(x)

    return (x, y, gene_symbols, index_to_genesymbol, genesymbol_to_index)