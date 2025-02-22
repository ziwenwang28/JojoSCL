import math
import torch
import torch.nn as nn

EPS = 1e-8

class InstanceLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    
    def generate_mask(self, batch_size):
        """Generate a binary mask that excludes self-similar and paired indices."""
        total = 2 * batch_size
        mask = torch.ones((total, total))
        mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask.bool()

    def forward(self, features):
        # Ensure we have the expected [batch_size, n_views, ...] dimensions.
        if features.ndim < 3:
            raise ValueError("Input 'features' must have at least 3 dimensions [batch, views, ...].")
        if features.ndim > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        device = torch.device('cuda:0') if features.is_cuda else torch.device('cpu')
        batch_size = features.shape[0]
        mask = self.generate_mask(batch_size).to(device)
        total = 2 * batch_size
        
        # Concatenate along the view dimension.
        concatenated = torch.cat(torch.unbind(features, dim=1), dim=0)
        sim_matrix = torch.matmul(concatenated, concatenated.T) / self.temperature

        # Extract the positive pairs from the diagonal offsets.
        pos_sim_i = torch.diag(sim_matrix, batch_size)
        pos_sim_j = torch.diag(sim_matrix, -batch_size)
        positives = torch.cat((pos_sim_i, pos_sim_j), dim=0).view(total, 1)
        negatives = sim_matrix[mask].view(total, -1)

        target = torch.zeros(total, device=positives.device, dtype=torch.long)
        logits = torch.cat((positives, negatives), dim=1)
        loss = self.criterion(logits, target) / total

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, num_classes, temperature):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mask = self.generate_cluster_mask(num_classes)

    def generate_cluster_mask(self, num_classes):
        """Create a mask for cluster-level comparisons to ignore trivial matches."""
        total = 2 * num_classes
        mask = torch.ones((total, total))
        mask.fill_diagonal_(0)
        for i in range(num_classes):
            mask[i, num_classes + i] = 0
            mask[num_classes + i, i] = 0
        return mask.bool()

    def forward(self, cluster_features):
        # Unpack cluster features from two views.
        c_i, c_j = torch.unbind(cluster_features, dim=1)
        
        # Normalize the summed probabilities.
        p_i = c_i.sum(dim=0).view(-1)
        p_i = p_i / p_i.sum()
        neg_entropy_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()

        p_j = c_j.sum(dim=0).view(-1)
        p_j = p_j / p_j.sum()
        neg_entropy_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()

        ne_loss = neg_entropy_i + neg_entropy_j

        total = 2 * self.num_classes
        concatenated_clusters = torch.cat(torch.unbind(cluster_features, dim=1), dim=0)
        sim_matrix = torch.matmul(concatenated_clusters, concatenated_clusters.T) / self.temperature

        pos_sim_i = torch.diag(sim_matrix, self.num_classes)
        pos_sim_j = torch.diag(sim_matrix, -self.num_classes)
        positives = torch.cat((pos_sim_i, pos_sim_j), dim=0).view(total, 1)
        negatives = sim_matrix[self.mask].view(total, -1)

        target = torch.zeros(total, device=positives.device, dtype=torch.long)
        logits = torch.cat((positives, negatives), dim=1)
        loss = self.criterion(logits, target) / total

        return loss + ne_loss
