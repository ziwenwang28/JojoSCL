# JojoSCL: Shrinkage Contrastive Learning for single-cell RNA Sequence Clustering


## Table of Contents
- [Overview](#overview)
- [Clustering Performance Comparison](#clustering-performance-comparison)
- [Installation](#installation)
  - [For GPU (CUDA-enabled)](#for-gpu-cuda-enabled)
  - [For CPU-only](#for-cpu-only)
- [Running JojoSCL](#running-jojoscl)
- [Data Preprocessing](#data-preprocessing)
- [Recommended Configurations of JojoSCL](#recommended-configurations-of-jojoscl)
- [Citation](#citation)


## Overview

In this paper, we present **JojoSCL: Shrinkage Contrastive Learning for single-cell RNA sequence Clustering**. Single-cell RNA sequencing (scRNA-seq) has revolutionized our understanding of cellular processes by enabling gene expression analysis at the individual cell level. However, the high dimensionality and inherent sparsity of scRNA-seq data present significant challenges for traditional clustering methods.

To address these issues, JojoSCL introduces a novel self-supervised contrastive learning framework that integrates a hierarchical Bayesian shrinkage estimator. By leveraging Stein’s Unbiased Risk Estimate (SURE) for optimization, our method refines both instance-level and cluster-level contrastive losses, leading to enhanced clustering performance across diverse scRNA-seq datasets.


<img src="https://github.com/user-attachments/assets/599844d4-c7a3-43f9-87db-34ddaa46c515" alt="MainFlowchart" width="600">

## Clustering Performance Comparison

*Clustering performance of different models across various datasets, evaluated over 10 consecutive runs in terms of ARI and NMI. The best clustering result for each dataset is **bolded** and the second-best result is <u>underlined</u>.*


| Dataset       | JojoSCL (ARI/NMI)        | Seurat (ARI/NMI)         | scziDesk (ARI/NMI)        | scDeepCluster (ARI/NMI)  | Contrastive-sc (ARI/NMI)  | ScCCL (ARI/NMI)          |
|---------------|--------------------------|--------------------------|---------------------------|--------------------------|---------------------------|--------------------------|
| **Adam**      | **0.9343/0.9191**        | 0.6806/0.7151            | 0.8273/0.8340             | 0.7892/0.7691            | 0.9034/0.8973             | <u>0.9133/0.9008</u>      |
| **Bladder**   | **0.6079/0.7507**        | 0.5825/0.6310            | 0.4907/0.6051             | <u>0.6030/0.7370</u>      | 0.5546/0.6704             | 0.5798/0.7332            |
| **Chen**      | **0.8168/0.7362**        | 0.5907/0.5563            | <u>0.7651/0.6413</u>       | 0.3791/0.3069            | <u>0.7224/0.6810</u>       | 0.7646/0.6802            |
| **Human brain** | **0.8905/0.8510**      | 0.7671/0.7315            | 0.8330/0.8328             | 0.8215/0.8007            | 0.8306/0.8179             | <u>0.8565/0.8340</u>      |
| **Klein**     | **0.8892/0.8547**        | 0.7436/0.7275            | <u>0.8014/0.7883</u>       | 0.7837/0.7512            | 0.6772/0.6559             | 0.7835/0.7745            |
| **Macosko**   | **0.8614/0.8145**        | 0.6335/0.7720            | 0.7252/0.8247             | 0.6209/0.7931            | 0.7762/0.7917             | <u>0.8581/0.7985</u>      |
| **Mouse**     | 0.6631/0.6995            | 0.6277/0.6641            | <u>0.7859/0.8013</u>       | **0.8177/0.8318**        | 0.7210/0.7554             | 0.6400/0.7033            |
| **Shekhar**   | **0.9624/0.8997**        | 0.7106/0.8377            | 0.5651/0.6426             | 0.6796/0.7995            | 0.7050/0.8341             | <u>0.9552/0.8860</u>      |
| **Yan**       | <u>0.8662/0.8793</u>     | 0.7095/0.7644            | <u>0.8665/0.8713</u>       | 0.8109/0.8663            | 0.8596/0.8710             | **0.8744/0.8813**        |
| **10X PBMC**  | **0.8080/0.8025**        | 0.5316/0.7129            | 0.6488/0.7366             | 0.7640/0.7580            | 0.7644/0.7569             | <u>0.7866/0.7782</u>      |
| **Average**   | 0.8300/0.8207            | 0.6577/0.7112            | 0.7309/0.7578             | 0.7070/0.7414            | 0.7514/0.7732             | 0.8012/0.7970            |



## Installation

Our minimal dependency set is as follows:

To install the required packages, choose one of the following options based on your system:

### For GPU (CUDA-enabled)
If you have a CUDA-enabled GPU, install PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### For CPU-only
If you are using a CPU-only system, install the CPU version of PyTorch:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -r requirements.txt
```





## Running JojoSCL

Our code supports both `.h5` and `.mat` file types. To run JojoSCL with your dataset, follow these steps:

1. **Prepare the Dataset File:**  
   Save the dataset file (e.g. `adam.mat` or `10X_PBMC.h5`) under the `data` folder.

2. **Configure the Dataset Type:**  
   In `main.py`, add the dataset name (without the extension) to the appropriate list:
   - For H5-format datasets, add the name to the `h5_datasets` list.
   - For MAT-format datasets, add the name to the `mat_datasets` list.

3. **Set Up Checkpoint Saving:**  
   Create a new folder inside the `save` directory named exactly after the dataset. This folder will be used to store checkpoints with the best results.

4. **Run the Code:**  
   Use the command line to run the script. For example, to run the `10X_PBMC` dataset, execute:
   
   ```bash
   python main.py --name 10X_PBMC
5. **Additional Configurations**:
   You may specify additional configurations as detailed in `config.py`.



## Data Preprocessing

The normalization and variable gene selection steps are implemented in `utils.py`. The process is as follows:

1. **Normalization:**  
   - The raw scRNA-seq expression data is rounded up and converted to 32-bit integers.
   - The data is encapsulated in an AnnData object.
   - Each cell's counts are normalized using size factors via `sc.pp.normalize_per_cell`.
   - A log-transformation is applied using `sc.pp.log1p` to stabilize the variance.

2. **Variable Gene Selection:**  
   - The top *n* variable genes are selected using Scanpy’s `sc.pp.highly_variable_genes` function with thresholds `min_mean=0.0125`, `max_mean=3`, and `min_disp=0.5`.
   - The number of genes retained is controlled by the `num_features` parameter (defaulting to 2000).

3. **Scaling:**  
   - Finally, the normalized data is scaled to standardize the expression values across genes, preparing it for downstream contrastive learning and clustering.

## Recommended Configurations of JojoSCL

Many settings for training and model configuration are managed in `config.py`. This includes parameters such as the dataset name, number of training epochs, number of genes to select (`--select_gene`), learning rate, dropout rate, batch size, and more. For example, the following parameters are defined:

- `--name`: Dataset name (default: "10X_PBMC")
- `--epoch`: Number of training epochs (default: 200)
- `--select_gene`: Number of genes to select (default: 2000)
- `--lr`: Learning rate (default: 0.2)
- `--dropout`: Dropout rate (default: 0.9)
- `--batch_size`: Batch size (default: 200)

These arguments allow for flexible configuration of the training process and are parsed using Python's `argparse` module. For contrastive learning, larger batch sizes tend to improve training stability. In our experiments, we use a batch size of 200 for moderate datasets such as `10X_PBMC` and `adam`, and a batch size of 2,000 for larger datasets such as `Macosko` and `Shekhar`.


## Citation

If you use this work in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{10821870,
  author={Wang, Ziwen},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 
  title={JojoSCL: Shrinkage Contrastive Learning for single-cell RNA sequence Clustering*}, 
  year={2024},
  volume={},
  number={},
  pages={2659-2666},
  keywords={Sequential analysis;RNA;Biological system modeling;Source coding;Estimation;Contrastive learning;Robustness;Bayes methods;Gene expression;Dispersion;ScRNA-seq Clustering;Contrastive Learning;Bayesian hierarchical modeling;Shrinkage estimator},
  doi={10.1109/BIBM62325.2024.10821870}
}



