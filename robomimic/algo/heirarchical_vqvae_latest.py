"""
Hierarchical VQ-VAE for Robot Action Tokenization
Implements two-level quantization: Actions → Subclusters (Z) → Clusters (Q)
Based on MSTCN architecture with dilated temporal convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F


class DilatedTemporalConvLayer(nn.Module):
    """
    Single dilated temporal convolution layer with residual connection
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.0
    ):
        super(DilatedTemporalConvLayer, self).__init__()

        # Padding to maintain sequence length
        self.padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, T] input tensor
        Returns:
            out: [B, C, T] output tensor
        """
        # Convolutional path
        out = self.conv(x)
        out = self.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)

        # Residual connection
        if self.residual is not None:
            x = self.residual(x)

        return out + x


class MSTCNStage(nn.Module):
    """
    Multi-Stage Temporal Convolutional Network Stage
    Contains multiple layers with exponentially increasing dilation
    """

    def __init__(
        self, num_layers, in_channels, out_channels, kernel_size=3, dropout=0.0
    ):
        super(MSTCNStage, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # First layer handles channel change
        self.layers.append(
            DilatedTemporalConvLayer(
                in_channels, out_channels, kernel_size, dilation=1, dropout=dropout
            )
        )

        # Remaining layers with increasing dilation: 2^0, 2^1, 2^2, ..., 2^(num_layers-1)
        for i in range(1, num_layers):
            dilation = 2**i
            self.layers.append(
                DilatedTemporalConvLayer(
                    out_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

    def forward(self, x):
        """
        Args:
            x: [B, C, T] input tensor
        Returns:
            out: [B, C, T] output tensor
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class ActionEncoder(nn.Module):
    """
    MSTCN-based encoder for robot actions
    Maps actions to latent embeddings
    """

    def __init__(
        self, action_dim, embed_dim, num_stages=2, num_layers_per_stage=10, dropout=0.1
    ):
        super(ActionEncoder, self).__init__()

        self.action_dim = action_dim
        self.embed_dim = embed_dim

        # Initial projection: action_dim -> embed_dim
        self.input_projection = nn.Linear(action_dim, embed_dim)

        # MSTCN stages
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            self.stages.append(
                MSTCNStage(
                    num_layers=num_layers_per_stage,
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    dropout=dropout,
                )
            )

        # Layer normalization for stability
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, actions):
        """
        Args:
            actions: [B, T, action_dim] action sequences
        Returns:
            embeddings: [B, T, embed_dim] encoded embeddings
        """
        # Project to embedding space
        x = self.input_projection(actions)  # [B, T, D]

        # Transpose for conv1d: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)

        # Pass through MSTCN stages
        for stage in self.stages:
            x = stage(x)

        # Transpose back: [B, D, T] -> [B, T, D]
        x = x.transpose(1, 2)

        # Layer normalization
        x = self.ln(x)

        return x


class ActionDecoder(nn.Module):
    """
    MSTCN-based decoder for robot actions
    Maps latent embeddings back to actions
    """

    def __init__(
        self, embed_dim, action_dim, num_stages=2, num_layers_per_stage=10, dropout=0.1
    ):
        super(ActionDecoder, self).__init__()

        self.embed_dim = embed_dim
        self.action_dim = action_dim

        # MSTCN stages
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            self.stages.append(
                MSTCNStage(
                    num_layers=num_layers_per_stage,
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    dropout=dropout,
                )
            )

        # Final projection: embed_dim -> action_dim
        self.output_projection = nn.Linear(embed_dim, action_dim)

    def forward(self, embeddings):
        """
        Args:
            embeddings: [B, T, embed_dim] latent embeddings
        Returns:
            actions: [B, T, action_dim] reconstructed actions
        """
        # Transpose for conv1d: [B, T, D] -> [B, D, T]
        x = embeddings.transpose(1, 2)

        # Pass through MSTCN stages
        for stage in self.stages:
            x = stage(x)

        # Transpose back: [B, D, T] -> [B, T, D]
        x = x.transpose(1, 2)

        # Project to action space
        actions = self.output_projection(x)  # [B, T, action_dim]

        return actions


class HierarchicalVQVAE(nn.Module):
    """
    Hierarchical Vector Quantized Variational Autoencoder
    Two-level quantization: Actions → Subclusters (Z) → Clusters (Q)
    """

    def __init__(
        self,
        action_dim,
        num_subclusters=256,
        num_clusters=128,
        embed_dim=512,
        num_stages=2,
        num_layers_per_stage=10,
        beta=0.8,
        dropout=0.1,
    ):
        super(HierarchicalVQVAE, self).__init__()

        self.action_dim = action_dim
        self.num_subclusters = num_subclusters
        self.num_clusters = num_clusters
        self.embed_dim = embed_dim
        self.beta = beta
        self.codebooks_initialized = False
        self.register_buffer("init_embeddings_buffer", torch.empty(0, embed_dim))

        # Encoder and Decoder
        self.encoder = ActionEncoder(
            action_dim=action_dim,
            embed_dim=embed_dim,
            num_stages=num_stages,
            num_layers_per_stage=num_layers_per_stage,
            dropout=dropout,
        )

        self.decoder = ActionDecoder(
            embed_dim=embed_dim,
            action_dim=action_dim,
            num_stages=num_stages,
            num_layers_per_stage=num_layers_per_stage,
            dropout=dropout,
        )

        # Codebooks (learnable embeddings)
        self.codebook_z = nn.Parameter(torch.randn(num_subclusters, embed_dim))
        self.codebook_q = nn.Parameter(torch.randn(num_clusters, embed_dim))

        # Initialize codebooks with uniform distribution
        nn.init.uniform_(self.codebook_z, -1.0 / num_subclusters, 1.0 / num_subclusters)
        nn.init.uniform_(self.codebook_q, -1.0 / num_clusters, 1.0 / num_clusters)
        # nn.init.normal_(self.codebook_z, mean=0.0, std=0.1)
        # nn.init.normal_(self.codebook_q, mean=0.0, std=0.1)

        # EMA tracking (not trained via backprop)
        self.register_buffer("cluster_size_z", torch.ones(num_subclusters))
        self.register_buffer("embed_avg_z", self.codebook_z.data.clone())
        self.register_buffer("cluster_size_q", torch.ones(num_clusters))
        self.register_buffer("embed_avg_q", self.codebook_q.data.clone())

    def quantize_to_subclusters(self, embeddings):
        """
        Quantize embeddings to subcluster codebook Z

        Args:
            embeddings: [B, T, D] encoder output
        Returns:
            quantized: [B, T, D] quantized vectors from codebook Z
            indices: [B, T] indices into codebook Z
            distances: [B, T, |Z|] cosine distances
        """
        B, T, D = embeddings.shape

        # L2 normalize embeddings and codebook
        embeddings_norm = F.normalize(embeddings, dim=-1)
        codebook_z_norm = F.normalize(self.codebook_z, dim=-1)

        # Flatten embeddings: [B, T, D] -> [B*T, D]
        embeddings_flat = embeddings_norm.reshape(-1, D)

        # Compute cosine similarity: [B*T, D] @ [D, |Z|] = [B*T, |Z|]
        similarities = torch.matmul(embeddings_flat, codebook_z_norm.t())

        # Find nearest prototype (max similarity)
        indices_flat = torch.argmax(similarities, dim=-1)  # [B*T]
        indices = indices_flat.reshape(B, T)  # [B, T]

        # Lookup quantized vectors
        quantized_flat = self.codebook_z[indices_flat]  # [B*T, D]
        quantized = quantized_flat.reshape(B, T, D)  # [B, T, D]

        # Straight-through estimator: pass gradients through unchanged
        quantized = embeddings + (quantized - embeddings).detach()  # removed

        # Distances (1 - similarity)
        distances = (1 - similarities).reshape(B, T, -1)

        return quantized, indices, distances

    def quantize_to_clusters(self, subclusters):
        """
        Quantize subclusters to cluster codebook Q

        Args:
            subclusters: [B, T, D] quantized vectors from Z
        Returns:
            quantized: [B, T, D] quantized vectors from codebook Q
            indices: [B, T] indices into codebook Q
            distances: [B, T, |Q|] cosine distances
        """
        B, T, D = subclusters.shape

        # L2 normalize
        subclusters_norm = F.normalize(subclusters, dim=-1)
        codebook_q_norm = F.normalize(self.codebook_q, dim=-1)

        # Flatten: [B, T, D] -> [B*T, D]
        subclusters_flat = subclusters_norm.reshape(-1, D)

        # Compute cosine similarity: [B*T, D] @ [D, |Q|] = [B*T, |Q|]
        similarities = torch.matmul(subclusters_flat, codebook_q_norm.t())

        # Find nearest prototype
        indices_flat = torch.argmax(similarities, dim=-1)  # [B*T]
        indices = indices_flat.reshape(B, T)  # [B, T]

        # Lookup quantized vectors
        quantized_flat = self.codebook_q[indices_flat]  # [B*T, D]
        quantized = quantized_flat.reshape(B, T, D)  # [B, T, D]

        # Straight-through estimator
        quantized = subclusters + (quantized - subclusters).detach()

        # Distances
        distances = (1 - similarities).reshape(B, T, -1)

        return quantized, indices, distances

    @torch.no_grad()
    def update_codebook_z(self, embeddings, z_indices):
        """
        Update subcluster codebook Z using exponential moving average
        Equation 3 from methodology
        """
        B, T, D = embeddings.shape

        # Flatten
        embeddings_flat = embeddings.reshape(-1, D)  # [B*T, D]
        indices_flat = z_indices.reshape(-1)  # [B*T]

        # One-hot encode indices: [B*T, |Z|]
        encodings = F.one_hot(indices_flat, self.num_subclusters).float()

        # Count assignments to each prototype
        # N_hat_zj = beta * N_zj + (1 - beta) * count
        counts = encodings.sum(0)  # [|Z|]
        self.cluster_size_z.mul_(self.beta).add_(counts, alpha=1 - self.beta)
        self.cluster_size_z.clamp_(min=1.0)  # new

        # Sum of embeddings assigned to each prototype
        # [|Z|, B*T] @ [B*T, D] = [|Z|, D]
        embed_sum = torch.matmul(encodings.t(), embeddings_flat)

        # Update embedding average
        self.embed_avg_z.mul_(self.beta).add_(embed_sum, alpha=1 - self.beta)

        # Update prototypes: z_hat_j = embed_avg_z / cluster_size_z
        # Add small epsilon to avoid division by zero
        n = self.cluster_size_z.unsqueeze(1)
        self.codebook_z.data.copy_(self.embed_avg_z / (n + 1e-5))

    @torch.no_grad()
    def update_codebook_q(self, subclusters, q_indices):
        """
        Update cluster codebook Q using exponential moving average
        Equation 4 from methodology
        """
        B, T, D = subclusters.shape

        # Flatten
        subclusters_flat = subclusters.reshape(-1, D)  # [B*T, D]
        indices_flat = q_indices.reshape(-1)  # [B*T]

        # One-hot encode: [B*T, |Q|]
        encodings = F.one_hot(indices_flat, self.num_clusters).float()

        # Count assignments
        counts = encodings.sum(0)  # [|Q|]
        self.cluster_size_q.mul_(self.beta).add_(counts, alpha=1 - self.beta)
        self.cluster_size_q.clamp_(min=0.5)

        # Sum of subclusters assigned to each cluster
        embed_sum = torch.matmul(encodings.t(), subclusters_flat)  # [|Q|, D]

        # Update embedding average
        self.embed_avg_q.mul_(self.beta).add_(embed_sum, alpha=1 - self.beta)

        # Update prototypes
        n = self.cluster_size_q.unsqueeze(1)
        self.codebook_q.data.copy_(self.embed_avg_q / (n + 1e-5))

    @torch.no_grad()
    def replace_dead_codes(self, embeddings, subclusters):
        """
        Replace unused prototypes with random embeddings from batch
        Threshold: N_hat_zj < 3 for Z, N_hat_qi < 1 for Q
        """
        # Check for dead codes in Z (subclusters)
        dead_z = self.cluster_size_z < 1
        if dead_z.any():
            num_dead = dead_z.sum().item()
            # Sample random embeddings from current batch
            B, T, D = embeddings.shape
            embeddings_flat = embeddings.reshape(-1, D)
            random_indices = torch.randint(
                0, B * T, (num_dead,), device=embeddings.device
            )

            replacement_z = embeddings_flat[random_indices]
            self.codebook_z.data[dead_z] = replacement_z
            self.cluster_size_z[dead_z] = 1.0
            self.embed_avg_z[dead_z] = replacement_z

        # Check for dead codes in Q (clusters)
        dead_q = self.cluster_size_q < 0.5
        if dead_q.any():
            num_dead = dead_q.sum().item()
            # Sample random subclusters from current batch
            B, T, D = subclusters.shape
            subclusters_flat = subclusters.reshape(-1, D)
            random_indices = torch.randint(
                0, B * T, (num_dead,), device=subclusters.device
            )

            replacement_q = subclusters_flat[random_indices]
            self.codebook_q.data[dead_q] = replacement_q
            self.cluster_size_q[dead_q] = 1.0
            self.embed_avg_q[dead_q] = replacement_q

    @torch.no_grad()
    def replace_dead_codes_old(self, embeddings, subclusters):
        """
        Replace unused prototypes with random embeddings from batch
        Threshold: N_hat_zj < 3 for Z, N_hat_qi < 1 for Q
        """
        # Check for dead codes in Z (subclusters)
        dead_z = self.cluster_size_z < 3
        if dead_z.any():
            num_dead = dead_z.sum().item()
            # Sample random embeddings from current batch
            B, T, D = embeddings.shape
            embeddings_flat = embeddings.reshape(-1, D)
            random_indices = torch.randint(
                0, B * T, (num_dead,), device=embeddings.device
            )
            self.codebook_z.data[dead_z] = embeddings_flat[random_indices]
            self.cluster_size_z[dead_z] = 1.0
            self.embed_avg_z[dead_z] = self.codebook_z.data[dead_z]

        # Check for dead codes in Q (clusters)
        dead_q = self.cluster_size_q < 1
        if dead_q.any():
            num_dead = dead_q.sum().item()
            # Sample random subclusters from current batch
            B, T, D = subclusters.shape
            subclusters_flat = subclusters.reshape(-1, D)
            random_indices = torch.randint(
                0, B * T, (num_dead,), device=subclusters.device
            )
            self.codebook_q.data[dead_q] = subclusters_flat[random_indices]
            self.cluster_size_q[dead_q] = 1.0
            self.embed_avg_q[dead_q] = self.codebook_q.data[dead_q]

    @torch.no_grad()
    def initialize_codebooks_with_current_batch(self, embeddings, num_samples=10000):
        """
        Initialize Z and Q codebooks using only the current batch embeddings.
        No accumulation of embeddings across batches.
        """
        device = embeddings.device
        B, T, D = embeddings.shape
        print("*" * 25)
        print("Embeddings shape", embeddings.shape)
        print("*" * 25)
        # Flatten embeddings of the current batch
        embeddings_flat = embeddings.reshape(-1, D).cpu()

        # Optional: sample to reduce KMeans cost
        if embeddings_flat.shape[0] > num_samples:
            idx = np.random.choice(embeddings_flat.shape[0], num_samples, replace=False)
            sample = embeddings_flat[idx].numpy()
        else:
            sample = embeddings_flat.numpy()

        print(f"[KMeans Init] Running KMeans on {sample.shape[0]} embeddings.")

        # ---------- KMeans for Z (subclusters) ----------
        kmeans_z = KMeans(
            n_clusters=self.num_subclusters, random_state=42, n_init=10, max_iter=300
        )
        kmeans_z.fit(sample)

        codebook_z_init = torch.from_numpy(kmeans_z.cluster_centers_).float().to(device)
        self.codebook_z.data.copy_(F.normalize(codebook_z_init, dim=-1))

        # ---------- KMeans for Q (coarse clusters) ----------
        kmeans_q = KMeans(
            n_clusters=self.num_clusters, random_state=42, n_init=10, max_iter=300
        )
        kmeans_q.fit(sample)

        codebook_q_init = torch.from_numpy(kmeans_q.cluster_centers_).float().to(device)
        self.codebook_q.data.copy_(F.normalize(codebook_q_init, dim=-1))

        print(
            f"[KMeans Init] Z ({self.num_subclusters}) and Q ({self.num_clusters}) codebooks initialized."
        )
        self.codebooks_initialized = True

    def forward(self, actions, training=True):
        """
        Full forward pass: encode → quantize → decode

        Args:
            actions: [B, T, action_dim] input action sequences
            training: bool, whether in training mode (updates codebooks)

        Returns:
            dict with keys:
                'embeddings': [B, T, D] encoder output
                'quantized_z': [B, T, D] subcluster quantized
                'quantized_q': [B, T, D] cluster quantized
                'reconstructed_actions': [B, T, action_dim]
                'z_indices': [B, T] subcluster indices
                'q_indices': [B, T] cluster indices
        """
        # 1. Encode actions to embeddings
        embeddings = self.encoder(actions)
        embeddings = F.normalize(embeddings, dim=-1)  # [B, T, D]

        if not self.codebooks_initialized:
            self.initialize_codebooks_with_current_batch(embeddings)

        # 2. Quantize to subclusters (Z)
        quantized_z, z_indices, z_distances = self.quantize_to_subclusters(embeddings)

        # 3. Quantize subclusters to clusters (Q)
        quantized_q, q_indices, q_distances = self.quantize_to_clusters(quantized_z)

        # 4. Decode from cluster level
        reconstructed = self.decoder(quantized_q)  # [B, T, action_dim]

        # 5. Update codebooks (if training)
        if training:
            self.update_codebook_z(embeddings, z_indices)
            self.update_codebook_q(quantized_z, q_indices)
            self.replace_dead_codes(embeddings, quantized_z)

        return {
            "embeddings": embeddings,
            "quantized_z": quantized_z,
            "quantized_q": quantized_q,
            "reconstructed_actions": reconstructed,
            "z_indices": z_indices,
            "q_indices": q_indices,
            "z_distances": z_distances,
            "q_distances": q_distances,
        }

    def encode(self, actions):
        """
        Encode actions without training (for inference)

        Args:
            actions: [B, T, action_dim]
        Returns:
            dict with encoded representations
        """
        with torch.no_grad():
            embeddings = self.encoder(actions)
            quantized_z, z_indices, _ = self.quantize_to_subclusters(embeddings)
            quantized_q, q_indices, _ = self.quantize_to_clusters(quantized_z)

        return {
            "embeddings": embeddings,
            "quantized_z": quantized_z,
            "quantized_q": quantized_q,
            "z_indices": z_indices,
            "q_indices": q_indices,
        }

    def compute_reconstruction_loss(self, pred_actions, true_actions):
        """
        L_rec = ||x - x_hat||^2
        """
        return F.mse_loss(pred_actions, true_actions)

    def compute_commitment_loss_z(self, embeddings, quantized_z):
        """
        L_commit_z = ||e_mt - sg[z_mt]||^2
        Push embeddings toward subclusters (stop gradient on quantized)
        """
        # return F.mse_loss(embeddings, quantized_z.detach())# old
        embeddings_norm = F.normalize(embeddings, dim=-1)
        quantized_z_norm = F.normalize(quantized_z, dim=-1)
        return F.mse_loss(embeddings_norm, quantized_z_norm.detach())

    def compute_commitment_loss_q(self, quantized_z, quantized_q):
        """
        L_commit_q = ||z_mt - sg[q_mt]||^2
        Push subclusters toward clusters (stop gradient on clusters)
        """
        # return F.mse_loss(quantized_z, quantized_q.detach())#old
        quantized_z_norm = F.normalize(quantized_z, dim=-1)
        quantized_q_norm = F.normalize(quantized_q, dim=-1)
        return F.mse_loss(quantized_z_norm, quantized_q_norm.detach())

    def compute_vqvae_loss(self, vqvae_outputs, true_actions, lambda_rec=1.0):
        """
        Total VQ-VAE loss: L = L_commit_z + L_commit_q + lambda_rec * L_rec

        Args:
            vqvae_outputs: dict from forward()
            true_actions: [B, T, action_dim] ground truth
            lambda_rec: weight for reconstruction loss

        Returns:
            dict of losses
        """
        L_rec = self.compute_reconstruction_loss(
            vqvae_outputs["reconstructed_actions"], true_actions
        )

        L_commit_z = self.compute_commitment_loss_z(
            vqvae_outputs["embeddings"], vqvae_outputs["quantized_z"]
        )

        L_commit_q = self.compute_commitment_loss_q(
            vqvae_outputs["quantized_z"], vqvae_outputs["quantized_q"]
        )

        total_loss = L_commit_z + L_commit_q + lambda_rec * L_rec

        return {
            "vqvae_loss": total_loss,
            "L_rec": L_rec,
            "L_commit_z": L_commit_z,
            "L_commit_q": L_commit_q,
        }

    def compute_soft_assignment(self, embeddings):
        """
        FIFA: Compute soft assignment probabilities using hierarchical similarity
        Equation 8: sim_mt,i = sum_j [(1 - d(e_mt, z_j)) + (1 - d(z_j, q_i))]

        Args:
            embeddings: [B, T, D] encoder output
        Returns:
            soft_probs: [B, T, |Q|] soft assignment probabilities to clusters
        """
        # Normalize
        embeddings_norm = F.normalize(embeddings, dim=-1)
        Z_norm = F.normalize(self.codebook_z, dim=-1)
        Q_norm = F.normalize(self.codebook_q, dim=-1)

        # Compute cosine similarities
        # sim(e_mt, z_j): [B, T, |Z|]
        sim_e_to_z = torch.matmul(embeddings_norm, Z_norm.t())

        # sim(z_j, q_i): [|Z|, |Q|]
        sim_z_to_q = torch.matmul(Z_norm, Q_norm.t())

        # Aggregate similarities for each cluster
        # For each cluster i: sum over all subclusters j of [sim(e, z_j) + sim(z_j, q_i)]
        # Expand dimensions for broadcasting
        sim_e_to_z_exp = sim_e_to_z.unsqueeze(-1)  # [B, T, |Z|, 1]
        sim_z_to_q_exp = sim_z_to_q.unsqueeze(0).unsqueeze(0)  # [1, 1, |Z|, |Q|]

        # Combined similarity: [B, T, |Z|, |Q|]
        sim_combined = sim_e_to_z_exp + sim_z_to_q_exp

        # Sum over subclusters: [B, T, |Q|]
        sim_mt_i = sim_combined.sum(dim=2)

        # Softmax for soft assignment
        soft_probs = F.softmax(sim_mt_i, dim=-1)

        return soft_probs

    def get_codebook_usage(self):
        """
        Get codebook utilization statistics

        Returns:
            dict with usage percentages
        """
        return {
            "z_utilization": (self.cluster_size_z > 0).float().mean().item(),
            "q_utilization": (self.cluster_size_q > 0).float().mean().item(),
            "z_dead_codes": (self.cluster_size_z < 3).sum().item(),
            "q_dead_codes": (self.cluster_size_q < 1).sum().item(),
        }
