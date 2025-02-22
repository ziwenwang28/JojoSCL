import os
import h5py
import torch
import config
import numpy as np
import train
import scipy.io as sio
from utils import data_preprocess

# Use the appropriate tensor type based on CUDA availability.
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Determine and display the computing device.
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running on device: {device}")

if __name__ == "__main__":
    # Define dataset categories based on file type.
    h5_datasets = ['10X_PBMC']
    mat_datasets = ['YAN']
    
    current_dir = os.getcwd()
    selected_dataset = config.args.name
    gene_expr = []
    true_labels = []
    
    if selected_dataset in h5_datasets:
        # For H5-format datasets, load the file.
        # Uncomment the following line to use a dynamic path:
        file_path = os.path.join(current_dir, "data", f"{selected_dataset}.h5")

        h5_file = h5py.File(file_path, 'r')
        gene_expr = np.array(h5_file.get('X'))
        true_labels = np.array(h5_file.get('Y')).reshape(-1)
        gene_expr = data_preprocess(gene_expr, config.args.select_gene)
    elif selected_dataset in mat_datasets:
        # For MAT-format datasets, load using scipy.io.
        # You can use the dynamic path or specify a fixed location.
        file_path = os.path.join(current_dir, "data", f"{selected_dataset}.mat")

        mat_file = sio.loadmat(file_path)
        gene_expr = np.array(mat_file['feature'])
        true_labels = np.array(mat_file['label']).reshape(-1)
        gene_expr = data_preprocess(gene_expr, config.args.select_gene)

    print(f"Gene expression matrix dimensions: {gene_expr.shape}")
    num_clusters = np.unique(true_labels).shape[0]
    print(f"Detected number of clusters: {num_clusters}")

    # Execute training.
    results = train.run(
        gene_exp=gene_expr,
        cluster_number=num_clusters,
        dataset=config.args.name,
        real_label=true_labels,
        epochs=config.args.epoch,
        lr=config.args.lr,
        temperature=config.args.temperature,
        dropout=config.args.dropout,
        layers=[config.args.enc_1, config.args.enc_2, config.args.enc_3, config.args.mlp_dim],
        save_pred=True,
        cluster_methods=config.args.cluster_methods,
        batch_size=config.args.batch_size,
        m=config.args.m,
        noise=config.args.noise
    )

    print("ACC:    ", results["acc"])
    print("ARI:    ", results["ari"])
    print("NMI:    ", results["nmi"])
    print("Time:   ", results["time"])
