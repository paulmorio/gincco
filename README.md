# gincco
Source code for GINCCo Paper "Unsupervised construction of computational graphs for gene expression data with explicit structural inductive biases"

If you are looking for the library of clustering algorithms also released for this paper, this can be found here: https://github.com/paulmorio/protclus

## Overview
Gene expression data is commonly used at the intersection of cancer research and machine learning for better understanding of the molecular status of tumour tissue. Deep learning predictive models have been employed for gene expression data due to their ability to scale and remove the need for manual feature engineering. However, gene expression data is often very high dimensional, noisy, and presented with a low number of samples. This poses significant problems for learning algorithms: models often overfit, learn noise, and struggle to capture biologically relevant information. 

GINCCo offers one method to tackle this issue by constructing computational graphs for the input gene expression data that incorporate prior knowledge embedded in the structure of gene interaction graphs such as protein-protein interaction networks. The structural construction of the computational graphs is driven by the use of topological clustering algorithms on protein-protein networks which incorporate inductive biases stemming from network biology research in protein complex discovery. Each of the entities inside of the computational graph constructed by GINCCo represent biological entities such as genes, candidate protein complexes, and phenotypes; allowing for introspective study of the learned functions. Furthermore it provides a biologically relevant mechanism for regularisation often yielding better predictive performance as we show in the paper.

## Prerequisites

- Python 3.6+
- Standard ML Packages as below

```bash
pip install numpy scipy pandas scikit-learn tqdm networkx
```

- PyTorch

If you have a CUDA enabled GPU, use the below command. Otherwise follow instructions on the PyTorch website.

```bash
pip install torch torchvision
```
## Citation
The paper is currently under review. If this code or its associated library was useful to you please consider citing the preprint (under an older title)

```
@misc{gincco,
      title={Incorporating network based protein complex discovery into automated model construction}, 
      author={Paul Scherer and Maja Trȩbacz and Nikola Simidjievski and Zohreh Shams and Helena Andres Terre and Pietro Liò and Mateja Jamnik},
      year={2020},
      eprint={2010.00387},
      archivePrefix={arXiv},
      primaryClass={q-bio.MN}
}
```

