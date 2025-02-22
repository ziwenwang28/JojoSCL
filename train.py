import os
import JojoSCL
import config
import time
import numpy as np
import torch
import contrastive_loss
from utils import update_learning_rate, store_model_checkpoint, compute_cluster_metrics
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run(gene_exp, cluster_number, dataset, real_label, epochs, lr, temperature, dropout, layers, batch_size, m,
        save_pred=True, noise=None, use_cpu=None, cluster_methods=None):
    if cluster_methods is None:
        cluster_methods = []
    results = {}

    start = time.time()
    embedding, best_model_epoch = train_model(
        gene_exp=gene_exp, 
        cluster_number=cluster_number, 
        real_label=real_label,
        epochs=epochs, 
        lr=lr, 
        temperature=temperature,
        dropout=dropout, 
        layers=layers, 
        batch_size=batch_size,
        m=m, 
        save_pred=save_pred, 
        noise=noise, 
        use_cpu=use_cpu
    )

    if save_pred:
        results["features"] = embedding
        results["max_epoch"] = best_model_epoch
    elapsed = time.time() - start
    res_eval = compute_cluster_metrics(
        embedding, 
        cluster_number, 
        real_label, 
        save_predictions=save_pred,
        clustering_methods=cluster_methods
    )
    results = {**results, **res_eval, "dataset": dataset, "time": elapsed}

    return results


def train_model(gene_exp, cluster_number, real_label, epochs, lr,
                temperature, dropout, layers, batch_size, m,
                save_pred=False, noise=None, use_cpu=None, evaluate_training=True):
    # Device selection
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dims = np.concatenate([[gene_exp.shape[1]], layers])
    
    # Initialize components from JojoSCL.py
    data_aug_model = JojoSCL.Augmenter(drop_rate=dropout)
    encoder_q = JojoSCL.EncoderBase(dims)
    encoder_k = JojoSCL.EncoderBase(dims)
    instance_projector = JojoSCL.ProjectionMLP(layers[2], layers[2] + layers[3], layers[2] + layers[3])
    cluster_projector = JojoSCL.ProjectionMLP(layers[2], layers[3], cluster_number)
    model = JojoSCL.JojoSCL(encoder_q, encoder_k, instance_projector, cluster_projector, cluster_number, m=m)
    
    data_aug_model.to(device)
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion_instance = contrastive_loss.InstanceLoss(temperature=temperature)
    criterion_cluster = contrastive_loss.ClusterLoss(cluster_number, temperature=temperature)

    max_value, best_model_epoch = -1, -1

    idx = np.arange(len(gene_exp))
    sure_params = []

    for epoch in range(epochs):
        model.train()
        update_learning_rate(optimizer, epoch, lr)
        np.random.shuffle(idx)
        loss_instance_total = 0
        loss_cluster_total = 0
        sure_loss_total = 0

        # Process mini-batches
        for pre_index in range(len(gene_exp) // batch_size + 1):
            c_idx = np.arange(pre_index * batch_size,
                              min(len(gene_exp), (pre_index + 1) * batch_size))
            if len(c_idx) == 0:
                continue
            c_idx = idx[c_idx]
            c_inp = gene_exp[c_idx]
            input1 = data_aug_model(torch.FloatTensor(c_inp))
            input2 = data_aug_model(torch.FloatTensor(c_inp))

            if noise is None or noise == 0:
                input1 = torch.FloatTensor(input1).to(device)
                input2 = torch.FloatTensor(input2).to(device)
            else:
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=input1.shape))
                input1 = torch.FloatTensor(input1 + noise_vec).to(device)
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=input2.shape))
                input2 = torch.FloatTensor(input2 + noise_vec).to(device)
            q_instance, q_cluster, k_instance, k_cluster = model(input1, input2)

            features_instance = torch.cat(
                [q_instance.unsqueeze(1), k_instance.unsqueeze(1)],
                dim=1
            )
            features_cluster = torch.cat(
                [q_cluster.t().unsqueeze(1), k_cluster.t().unsqueeze(1)],
                dim=1
            )
            loss_instance = criterion_instance(features_instance)
            loss_cluster = criterion_cluster(features_cluster)

            # ---- Compute LSURE loss as per the paper's formulation ----
            # Move instance embeddings to CPU and cluster using KMeans
            q_instance_cpu = q_instance.cpu().detach().numpy()  # shape: (n_samples, P)
            kmeans = KMeans(n_clusters=cluster_number, random_state=0).fit(q_instance_cpu)
            labels = kmeans.labels_
            P = q_instance_cpu.shape[1]
            LSURE_batch = 0.0
            for k in range(cluster_number):
                idx_k = np.where(labels == k)[0]
                if len(idx_k) < 2:
                    continue  # Skip clusters with too few samples
                # Compute cluster centroid mu_k
                mu_k = np.mean(q_instance_cpu[idx_k], axis=0)
                # Compute squared errors for samples in cluster k
                diff = q_instance_cpu[idx_k] - mu_k
                squared_errors = np.sum(diff ** 2, axis=1)
                # Estimate intra-cluster variance sigma2_k (averaged over dimensions)
                variances = np.var(q_instance_cpu[idx_k], axis=0, ddof=1)
                sigma2_k = np.mean(variances)
                N_k = len(idx_k)
                # Estimate tau2_k using the central limit theorem: tau2_k = sigma2_k / N_k
                tau2_k = sigma2_k / N_k
                # Compute shrinkage factor
                denom = tau2_k + sigma2_k
                factor = sigma2_k / denom if denom != 0 else 0
                # Sum the SURE contributions for all samples in cluster k
                LSURE_cluster = np.sum(factor * (squared_errors + P * (tau2_k - sigma2_k)))
                LSURE_batch += LSURE_cluster
            sure_loss = LSURE_batch
            # ------------------------------------------------------------
            
            sure_loss_total += sure_loss
            total_loss = loss_instance + loss_cluster + sure_loss
            loss_instance_total += loss_instance.item()
            loss_cluster_total += loss_cluster.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            sure_params.append((epoch, pre_index, sure_loss))

        if evaluate_training and real_label is not None:
            model.eval()
            with torch.no_grad():
                q_instance, _, _, _ = model(torch.FloatTensor(gene_exp).to(device), None)
                features = q_instance.detach().cpu().numpy()
            res = compute_cluster_metrics(features, cluster_number, real_label, save_predictions=save_pred)
            print(
                f"Epoch {epoch}: Loss: {loss_instance_total + loss_cluster_total + sure_loss_total}, ACC: {res['acc']}, ARI: {res['ari']}, "
                f"NMI: {res['nmi']} "
            )

            if res['ari'] + res['nmi'] >= max_value:
                max_value = res['ari'] + res['nmi']
                store_model_checkpoint(config.args.name, model, optimizer, epoch, best_model_epoch)
                best_model_epoch = epoch

    # Identify the epoch and batch index with the lowest LSURE loss
    best_sure_idx = np.argmin([param[2] for param in sure_params])
    best_sure_epoch, best_sure_index, _ = sure_params[best_sure_idx]

    # Fine-tuning on the mini-batch with the lowest LSURE loss
    if best_model_epoch != -1:
        model_fp = os.path.join(os.getcwd(), 'save', config.args.name, f"checkpoint_{best_model_epoch}.tar")
        model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
        model.to(device)

        update_learning_rate(optimizer, best_sure_epoch, lr)
        model.train()
        for pre_index in range(len(gene_exp) // batch_size + 1):
            if pre_index != best_sure_index:
                continue
            c_idx = np.arange(pre_index * batch_size, min(len(gene_exp), (pre_index + 1) * batch_size))
            if len(c_idx) == 0:
                continue
            c_idx = idx[c_idx]
            c_inp = gene_exp[c_idx]
            input1 = data_aug_model(torch.FloatTensor(c_inp))
            input2 = data_aug_model(torch.FloatTensor(c_inp))

            if noise is None or noise == 0:
                input1 = torch.FloatTensor(input1).to(device)
                input2 = torch.FloatTensor(input2).to(device)
            else:
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=input1.shape))
                input1 = torch.FloatTensor(input1 + noise_vec).to(device)
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=input2.shape))
                input2 = torch.FloatTensor(input2 + noise_vec).to(device)

            q_instance, q_cluster, k_instance, k_cluster = model(input1, input2)

            features_instance = torch.cat(
                [q_instance.unsqueeze(1), k_instance.unsqueeze(1)],
                dim=1
            )
            features_cluster = torch.cat(
                [q_cluster.t().unsqueeze(1), k_cluster.t().unsqueeze(1)],
                dim=1
            )
            loss_instance = criterion_instance(features_instance)
            loss_cluster = criterion_cluster(features_cluster)

            q_instance_cpu = q_instance.cpu().detach().numpy()
            kmeans = KMeans(n_clusters=cluster_number, random_state=0).fit(q_instance_cpu)
            labels = kmeans.labels_
            P = q_instance_cpu.shape[1]
            LSURE_batch = 0.0
            for k in range(cluster_number):
                idx_k = np.where(labels == k)[0]
                if len(idx_k) < 2:
                    continue
                mu_k = np.mean(q_instance_cpu[idx_k], axis=0)
                diff = q_instance_cpu[idx_k] - mu_k
                squared_errors = np.sum(diff ** 2, axis=1)
                variances = np.var(q_instance_cpu[idx_k], axis=0, ddof=1)
                sigma2_k = np.mean(variances)
                N_k = len(idx_k)
                tau2_k = sigma2_k / N_k
                denom = tau2_k + sigma2_k
                factor = sigma2_k / denom if denom != 0 else 0
                LSURE_cluster = np.sum(factor * (squared_errors + P * (tau2_k - sigma2_k)))
                LSURE_batch += LSURE_cluster
            sure_loss = LSURE_batch

            total_loss = loss_instance + loss_cluster + sure_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    model.eval()
    model_fp = os.path.join(os.getcwd(), 'save', config.args.name, f"checkpoint_{best_model_epoch}.tar")
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)
    print(f"Model is on device: {next(model.parameters()).device}")
    with torch.no_grad():
        q_instance, _, _, _ = model(torch.FloatTensor(gene_exp).to(device), None)
        features = q_instance.detach().cpu().numpy()

    return features, best_model_epoch
