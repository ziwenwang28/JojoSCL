import os
import time
import math
import torch
import numpy as np
import scanpy as sc
import scipy as sp
from scipy.optimize import linear_sum_assignment

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def data_preprocess(raw_expr, num_features):
    """
    Rounds up raw gene expression values, converts them to int32,
    builds an AnnData object, applies normalization and feature selection,
    and returns the processed matrix as float32.
    """
    rounded_expr = np.ceil(raw_expr).astype(np.int32)
    print(rounded_expr.shape, rounded_expr.shape, f"keeping {num_features} features")
    adata_obj = sc.AnnData(rounded_expr)
    adata_obj = normalize_counts(
        adata_obj,
        copy_flag=True,
        highly_genes=num_features,
        size_factors=True,
        normalize_input=True,
        logtrans_input=True
    )
    return adata_obj.X.astype(np.float32)


def normalize_counts(adata_obj, copy_flag=True, highly_genes=None, filter_low_counts=True,
                     size_factors=True, normalize_input=True, logtrans_input=True):
    """
    Normalizes count data in an AnnData object. This function filters out low-count genes and cells,
    applies per-cell normalization, log-transformation, variable gene selection, and scaling.
    """
    if isinstance(adata_obj, sc.AnnData):
        adata_obj = adata_obj.copy() if copy_flag else adata_obj
    elif isinstance(adata_obj, str):
        adata_obj = sc.read(adata_obj)
    else:
        raise NotImplementedError("Input must be an AnnData object or a valid file path.")

    error_msg = 'Ensure that adata_obj.X contains raw count data.'
    assert 'n_count' not in adata_obj.obs, error_msg
    if adata_obj.X.size < 50e6:
        if sp.sparse.issparse(adata_obj.X):
            assert (adata_obj.X.astype(int) != adata_obj.X).nnz == 0, error_msg
        else:
            assert np.all(adata_obj.X.astype(int) == adata_obj.X), error_msg

    if filter_low_counts:
        sc.pp.filter_genes(adata_obj, min_counts=1)
        sc.pp.filter_cells(adata_obj, min_counts=1)

    adata_obj.raw = adata_obj.copy() if (size_factors or normalize_input or logtrans_input) else adata_obj

    if size_factors:
        adata_obj.X = adata_obj.X.astype(float)
        sc.pp.normalize_per_cell(adata_obj)
        adata_obj.obs['size_factors'] = adata_obj.obs.n_counts / np.median(adata_obj.obs.n_counts)
    else:
        adata_obj.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata_obj)

    if highly_genes is not None:
        sc.pp.highly_variable_genes(
            adata_obj,
            min_mean=0.0125,
            max_mean=3,
            min_disp=0.5,
            n_top_genes=highly_genes,
            subset=True
        )

    if normalize_input:
        sc.pp.scale(adata_obj)

    return adata_obj


def select_device(force_cpu=None):
    """
    Determines the computing device to use. If force_cpu is None and a GPU is available,
    returns the CUDA device; otherwise, returns CPU.
    """
    if force_cpu is None and torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def update_learning_rate(opt_obj, current_epoch, base_lr):
    """
    Adjusts the learning rate in the optimizer using a cosine annealing schedule.
    The learning rate is updated in place.
    """
    scheduler_config = {
        'total_epochs': 500,
        'mode': 'cosine',
        'kwargs': {'lr_decay_rate': 0.1},
    }

    if scheduler_config['mode'] == 'cosine':
        min_lr = base_lr * (scheduler_config['kwargs']['lr_decay_rate'] ** 3)
        new_lr = min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * current_epoch / scheduler_config['total_epochs'])) / 2
    elif scheduler_config['mode'] == 'step':
        decay_epochs = scheduler_config['kwargs'].get('lr_decay_epochs', [])
        steps = np.sum(current_epoch > np.array(decay_epochs))
        new_lr = base_lr * (scheduler_config['kwargs']['lr_decay_rate'] ** steps) if steps > 0 else base_lr
    elif scheduler_config['mode'] == 'constant':
        new_lr = base_lr
    else:
        raise ValueError('Invalid learning rate schedule specified.')

    for group in opt_obj.param_groups:
        group['lr'] = new_lr

    return base_lr


def store_model_checkpoint(model_id, net_model, opt_obj, curr_epoch, prev_epoch):
    """
    Saves the current state of the model. If a previous checkpoint exists, it is removed.
    """
    save_folder = os.path.join(os.getcwd(), "save", model_id)
    if prev_epoch != -1:
        old_checkpoint = os.path.join(save_folder, f"checkpoint_{prev_epoch}.tar")
        if os.path.exists(old_checkpoint):
            os.remove(old_checkpoint)
    checkpoint_file = os.path.join(save_folder, f"checkpoint_{curr_epoch}.tar")
    checkpoint_data = {
        'net': net_model.state_dict(),
        'optimizer': opt_obj.state_dict(),
        'epoch': curr_epoch
    }
    torch.save(checkpoint_data, checkpoint_file)


def calculate_clustering_accuracy(true_vals, pred_vals):
    """
    Calculates clustering accuracy by using the Hungarian algorithm to match predicted labels
    with true labels.
    """
    true_vals = true_vals.astype(np.int64)
    pred_vals = pred_vals.astype(np.int64)
    num_classes = max(pred_vals.max(), true_vals.max()) + 1
    cost_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for idx in range(pred_vals.size):
        cost_matrix[pred_vals[idx], true_vals[idx]] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    accuracy = cost_matrix[row_ind, col_ind].sum() / pred_vals.size
    return accuracy


def compute_cluster_metrics(embedded_data, num_clusters, true_vals, save_predictions=False, clustering_methods=None):
    """
    Performs clustering on the provided embeddings (default: KMeans) and computes ARI, NMI,
    and clustering accuracy (ACC) if ground truth labels are provided.
    """
    if clustering_methods is None:
        clustering_methods = ["KMeans"]

    metrics = {"cluster_start_time": time.time()}

    if "KMeans" in clustering_methods:
        kmeans_model = KMeans(n_clusters=num_clusters, init="k-means++", random_state=0)
        predicted_labels = kmeans_model.fit_predict(embedded_data)
        if true_vals is not None:
            ari_score = adjusted_rand_score(true_vals, predicted_labels)
            nmi_score = normalized_mutual_info_score(true_vals, predicted_labels)
            acc_score = calculate_clustering_accuracy(true_vals, predicted_labels)
            metrics["ari"] = round(ari_score, 4)
            metrics["nmi"] = round(nmi_score, 4)
            metrics["acc"] = round(acc_score, 4)
        metrics["kmeans_end_time"] = time.time()
        if save_predictions:
            metrics["predicted_labels"] = predicted_labels

    return metrics
