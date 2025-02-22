import torch
import torch.nn as nn
import torch.nn.functional as F

class Augmenter(nn.Module):
    def __init__(self, drop_rate=0.9):
        super().__init__()
        self.dropout_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        # Apply dropout as a simple data augmentation.
        return self.dropout_layer(x)


class EncoderBase(nn.Module):
    def __init__(self, dims):
        """
        Constructs a feed-forward network based on the provided dimensions.
        dims: list of integers; expected format: [input_dim, hidden1, hidden2, output_dim]
        """
        super().__init__()
        self.dims = dims
        # Build the network: Linear -> BatchNorm -> ReLU repeated three times.
        self.net = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.BatchNorm1d(dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(dims[1], dims[2]),
            nn.BatchNorm1d(dims[2]),
            nn.ReLU(inplace=True),
            nn.Linear(dims[2], dims[3]),
            nn.BatchNorm1d(dims[3]),
            nn.ReLU(inplace=True)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        latent = self.net(x)
        # Normalize the latent representation.
        return F.normalize(latent, dim=1)


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        """
        A simple MLP projection head.
        in_dim: dimension of input features.
        hidden_dim: dimension of the hidden layer.
        out_dim: desired output dimension.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class JojoSCL(nn.Module):
    def __init__(self, encoder_q, encoder_k, inst_proj, clust_proj, num_classes, m=0.2):
        """
        Main model for self-supervised clustering.
        encoder_q: query encoder.
        encoder_k: key encoder (initialized with encoder_q weights).
        inst_proj: projector for instance-level features.
        clust_proj: projector for clustering features.
        num_classes: expected number of clusters.
        m: momentum coefficient.
        """
        super().__init__()
        self.num_classes = num_classes
        self.m = m

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        # Initialize encoder_k with encoder_q parameters and freeze them.
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.inst_proj = inst_proj
        self.clust_proj = nn.Sequential(
            clust_proj,
            nn.Softmax(dim=1)
        )

    @torch.no_grad()
    def update_key_encoder(self):
        """Momentum update for the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    def forward(self, query_input, key_input):
        # Process query branch.
        q = self.encoder_q(query_input)
        q_instance = F.normalize(self.inst_proj(q), dim=1)
        q_cluster = self.clust_proj(q)

        if key_input is None:
            return q_instance, q_cluster, None, None

        # Process key branch with momentum update.
        with torch.no_grad():
            self.update_key_encoder()
            k = self.encoder_k(key_input)
            k_instance = F.normalize(self.inst_proj(k), dim=1)
            k_cluster = self.clust_proj(k)

        return q_instance, q_cluster, k_instance, k_cluster
