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
        # self.codebooks_initialized = False
        self.kmeans_buffer = []  # list of tensors [N, D]
        self.kmeans_sample_cap = 200_000  # max samples to accumulate
        self.kmeans_min_samples = max(
            self.num_subclusters * 10,  # need 10x samples per cluster
            self.num_clusters * 10,
        )

        self.codebooks_initialized = False

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

        # Codebooks (learnable embeddings) - initialize with small random values
        self.codebook_z = nn.Parameter(torch.randn(num_subclusters, embed_dim) * 0.01)
        self.codebook_q = nn.Parameter(torch.randn(num_clusters, embed_dim) * 0.01)

        # EMA tracking (not trained via backprop)
        self.register_buffer("cluster_size_z", torch.ones(num_subclusters))
        self.register_buffer("embed_avg_z", self.codebook_z.data.clone())
        self.register_buffer("cluster_size_q", torch.ones(num_clusters))
        self.register_buffer("embed_avg_q", self.codebook_q.data.clone())

    def initialize_codebooks_with_kmeans(self, embeddings, num_samples=10000):
        """
        Initialize codebooks using K-means clustering on actual embeddings.
        Call this once with real data before training.

        Args:
            embeddings: [B, T, D] tensor of encoded embeddings
            num_samples: number of samples to use for K-means (for efficiency)
        """
        if self.codebooks_initialized:
            print("Codebooks already initialized, skipping K-means init")
            return

        print("Initializing codebooks with K-means...")

        # Get device
        device = embeddings.device

        # Flatten and move to CPU for sklearn
        B, T, D = embeddings.shape
        embeddings_flat = embeddings.reshape(-1, D).detach().cpu().numpy()

        print(f"  Total embeddings available: {len(embeddings_flat)}")

        # Sample for efficiency if needed
        if len(embeddings_flat) > num_samples:
            print(f"  Sampling {num_samples} embeddings for K-means")
            indices = np.random.choice(len(embeddings_flat), num_samples, replace=False)
            embeddings_sample = embeddings_flat[indices]
        else:
            embeddings_sample = embeddings_flat

        # ===== K-means for Z (subclusters) =====
        print(f"  K-means for Z codebook ({self.num_subclusters} clusters)...")
        try:
            kmeans_z = KMeans(
                n_clusters=self.num_subclusters,
                random_state=42,
                n_init=10,
                max_iter=300,
                verbose=0,
            )
            kmeans_z.fit(embeddings_sample)
            codebook_z_init = torch.from_numpy(kmeans_z.cluster_centers_).float()

            # Normalize to match quantization space
            codebook_z_init = F.normalize(codebook_z_init, dim=-1)
            self.codebook_z.data = codebook_z_init.to(device)

            print(
                f"    ✓ Z codebook initialized (centers norm: {codebook_z_init.norm(dim=1).mean():.4f})"
            )
        except Exception as e:
            print(f"    ✗ K-means failed for Z: {e}")
            return

        # ===== K-means for Q (clusters) =====
        print(f"  K-means for Q codebook ({self.num_clusters} clusters)...")
        try:
            kmeans_q = KMeans(
                n_clusters=self.num_clusters,
                random_state=42,
                n_init=10,
                max_iter=300,
                verbose=0,
            )
            kmeans_q.fit(embeddings_sample)
            codebook_q_init = torch.from_numpy(kmeans_q.cluster_centers_).float()

            # Normalize to match quantization space
            codebook_q_init = F.normalize(codebook_q_init, dim=-1)
            self.codebook_q.data = codebook_q_init.to(device)

            print(
                f"    ✓ Q codebook initialized (centers norm: {codebook_q_init.norm(dim=1).mean():.4f})"
            )
        except Exception as e:
            print(f"    ✗ K-means failed for Q: {e}")
            return

        # Reset EMA tracking with new codebooks
        self.embed_avg_z.data = self.codebook_z.data.clone()
        self.embed_avg_q.data = self.codebook_q.data.clone()
        self.cluster_size_z.fill_(1.0)
        self.cluster_size_q.fill_(1.0)

        self.codebooks_initialized = True
        print("✓ Codebooks initialized successfully with K-means!")

    def quantize_to_subclusters(self, embeddings):
        """Quantize with consistent normalization"""
        B, T, D = embeddings.shape

        # Normalize ONCE
        embeddings_norm = F.normalize(embeddings, dim=-1)
        codebook_z_norm = F.normalize(self.codebook_z, dim=-1)

        embeddings_flat = embeddings_norm.reshape(-1, D)

        # Compute similarity
        similarities = torch.matmul(embeddings_flat, codebook_z_norm.t())
        indices_flat = torch.argmax(similarities, dim=-1)
        indices = indices_flat.reshape(B, T)

        # Return NORMALIZED quantized vectors
        quantized_flat = codebook_z_norm[indices_flat]  # ← NORMALIZED!
        quantized = quantized_flat.reshape(B, T, D)

        distances = (1 - similarities).reshape(B, T, -1)

        return quantized, indices, distances

    def quantize_to_clusters(self, subclusters):
        """Same for Q"""
        B, T, D = subclusters.shape

        subclusters_norm = F.normalize(subclusters, dim=-1)
        codebook_q_norm = F.normalize(self.codebook_q, dim=-1)

        subclusters_flat = subclusters_norm.reshape(-1, D)

        similarities = torch.matmul(subclusters_flat, codebook_q_norm.t())
        indices_flat = torch.argmax(similarities, dim=-1)
        indices = indices_flat.reshape(B, T)

        quantized_flat = codebook_q_norm[indices_flat]  # ← NORMALIZED!
        quantized = quantized_flat.reshape(B, T, D)

        distances = (1 - similarities).reshape(B, T, -1)

        return quantized, indices, distances

    @torch.no_grad()
    def update_codebook_z(self, embeddings, z_indices):
        """Update using NORMALIZED embeddings"""
        B, T, D = embeddings.shape

        # Normalize embeddings before EMA update
        embeddings_norm = F.normalize(embeddings, dim=-1)
        embeddings_flat = embeddings_norm.reshape(-1, D)  # ← NORMALIZED
        indices_flat = z_indices.reshape(-1)

        encodings = F.one_hot(indices_flat, self.num_subclusters).float()
        counts = encodings.sum(0)

        self.cluster_size_z.mul_(self.beta).add_(counts, alpha=1 - self.beta)

        # EMA update with normalized embeddings
        embed_sum = torch.matmul(encodings.t(), embeddings_flat)
        self.embed_avg_z.mul_(self.beta).add_(embed_sum, alpha=1 - self.beta)

        # Normalize codebook
        n = self.cluster_size_z.unsqueeze(1)
        codebook_unnormalized = self.embed_avg_z / (n + 1e-5)
        self.codebook_z.data = F.normalize(codebook_unnormalized, dim=-1)

    @torch.no_grad()
    def update_codebook_q(self, subclusters, q_indices):
        """Update using NORMALIZED subclusters"""
        B, T, D = subclusters.shape

        # Normalize before EMA update
        subclusters_norm = F.normalize(subclusters, dim=-1)
        subclusters_flat = subclusters_norm.reshape(-1, D)
        indices_flat = q_indices.reshape(-1)

        encodings = F.one_hot(indices_flat, self.num_clusters).float()
        counts = encodings.sum(0)

        self.cluster_size_q.mul_(self.beta).add_(counts, alpha=1 - self.beta)

        embed_sum = torch.matmul(encodings.t(), subclusters_flat)
        self.embed_avg_q.mul_(self.beta).add_(embed_sum, alpha=1 - self.beta)

        # Normalize codebook
        n = self.cluster_size_q.unsqueeze(1)
        codebook_unnormalized = self.embed_avg_q / (n + 1e-5)
        self.codebook_q.data = F.normalize(codebook_unnormalized, dim=-1)

    @torch.no_grad()
    def replace_dead_codes(self, embeddings, subclusters):

        dead_z = self.cluster_size_z < 1.0
        if dead_z.any():
            B, T, D = embeddings.shape
            embeddings_flat = embeddings.reshape(-1, D)
            random_indices = torch.randint(
                0, B * T, (dead_z.sum().item(),), device=embeddings.device
            )

            # Normalize replacement
            replacement_z = F.normalize(embeddings_flat[random_indices], dim=-1)
            self.codebook_z.data[dead_z] = replacement_z
            self.cluster_size_z[dead_z] = 1.0
            self.embed_avg_z[dead_z] = replacement_z

        dead_q = self.cluster_size_q < 0.5
        if dead_q.any():
            B, T, D = subclusters.shape
            subclusters_flat = subclusters.reshape(-1, D)
            random_indices = torch.randint(
                0, B * T, (dead_q.sum().item(),), device=subclusters.device
            )

            replacement_q = F.normalize(subclusters_flat[random_indices], dim=-1)
            self.codebook_q.data[dead_q] = replacement_q
            self.cluster_size_q[dead_q] = 1.0
            self.embed_avg_q[dead_q] = replacement_q

    def accumulate_embeddings_for_kmeans(self, embeddings):
        """
        Store embeddings from multiple batches for KMeans init.
        embeddings: [B, T, D]
        """
        flat = embeddings.reshape(-1, embeddings.size(-1)).detach().cpu()
        self.kmeans_buffer.append(flat)

        # Limit to memory cap
        total = sum(x.size(0) for x in self.kmeans_buffer)
        while total > self.kmeans_sample_cap:
            removed = self.kmeans_buffer.pop(0)
            total -= removed.size(0)

        print(f"[KMeans Buffer] {total} samples stored")

    def try_initialize_codebooks(self, device):
        """
        Initializes BOTH Z and Y codebooks using accumulated embeddings.
        Runs once only when enough samples exist.
        """
        if self.codebooks_initialized:
            return

        # Count samples
        total = sum(x.size(0) for x in self.kmeans_buffer)

        if total < self.kmeans_min_samples:
            print(
                f"[KMeans Init] Not enough samples yet "
                f"({total}/{self.kmeans_min_samples})"
            )
            return

        print(f"[KMeans Init] Running Z and Y KMeans on {total} samples...")

        # Merge buffer into a single [N, D]
        all_samples = torch.cat(self.kmeans_buffer, dim=0)
        np_samples = all_samples.numpy()

        # Downsample to cap
        if len(np_samples) > self.kmeans_sample_cap:
            idx = np.random.choice(
                len(np_samples), self.kmeans_sample_cap, replace=False
            )
            np_samples = np_samples[idx]

        D = np_samples.shape[1]

        from sklearn.cluster import KMeans

        # -------------------------------
        # 1) Fine Codebook: Z (subclusters)
        # -------------------------------
        print(f"  KMeans Z: {self.num_subclusters} clusters")

        n_z = self.num_subclusters
        usable_z = min(n_z, len(np_samples))

        kmeans_z = KMeans(n_clusters=usable_z, random_state=42, n_init=10, max_iter=300)
        kmeans_z.fit(np_samples)

        centers_z = torch.from_numpy(kmeans_z.cluster_centers_).float()
        centers_z = F.normalize(centers_z, dim=-1)

        # If fewer samples than clusters → pad randomly
        if usable_z < n_z:
            missing = n_z - usable_z
            print(f"    Padding Z with {missing} random centers")
            rand = torch.randn(missing, D)
            rand = F.normalize(rand, dim=-1)
            centers_z = torch.cat([centers_z, rand], dim=0)

        self.codebook_z.data = centers_z.to(device)
        print("    ✓ Z codebook initialized")

        # -------------------------------
        # 2) Coarse Codebook: Y (clusters over Z centers)
        # -------------------------------
        print(f"  KMeans Y: {self.num_clusters} clusters")

        n_y = self.num_clusters

        # Use Z centers as input to Y-level KMeans (hierarchical)
        z_centers_np = centers_z.cpu().numpy()

        usable_y = min(n_y, len(z_centers_np))

        kmeans_y = KMeans(n_clusters=usable_y, random_state=42, n_init=10, max_iter=300)
        kmeans_y.fit(z_centers_np)

        centers_y = torch.from_numpy(kmeans_y.cluster_centers_).float()
        centers_y = F.normalize(centers_y, dim=-1)

        # Pad if needed
        if usable_y < n_y:
            missing = n_y - usable_y
            print(f"    Padding Y with {missing} random centers")
            rand = torch.randn(missing, D)
            rand = F.normalize(rand, dim=-1)
            centers_y = torch.cat([centers_y, rand], dim=0)

        self.codebook_y.data = centers_y.to(device)
        print("    ✓ Y codebook initialized")

        # Mark finished
        self.codebooks_initialized = True
        print("  ✓ Hierarchical codebook initialization complete!")

    def forward(self, actions, training=True):
        """
        Full forward pass: encode → quantize → decode

        Always returns the full dict, even if codebooks are not yet initialized.
        """

        # --------------------------------------------------------
        # 1. Encode actions → embeddings
        # --------------------------------------------------------
        embeddings = self.encoder(actions)  # [B, T, D]

        # ========================================================
        # 2. Multi-batch hierarchical KMeans initialization (Z + Q)
        # ========================================================
        if not self.codebooks_initialized:

            # accumulate this batch
            self.accumulate_embeddings_for_kmeans(embeddings)

            # try to initialize both codebooks
            self.try_initialize_codebooks(device=embeddings.device)

            # If NOT initialized → return "identity forward"
            if not self.codebooks_initialized:
                # NOTE: all keys must exist to avoid breaking training loop
                return {
                    "embeddings": embeddings,
                    "quantized_z": embeddings,  # fallback: no quantization yet
                    "quantized_q": embeddings,  # fallback: identity mapping
                    "reconstructed_actions": self.decoder(embeddings),
                    "z_indices": torch.zeros(
                        embeddings.shape[:2], dtype=torch.long, device=embeddings.device
                    ),
                    "q_indices": torch.zeros(
                        embeddings.shape[:2], dtype=torch.long, device=embeddings.device
                    ),
                    "z_distances": None,
                    "q_distances": None,
                }

        # --------------------------------------------------------
        # 3. Quantize to subclusters (Z)
        # --------------------------------------------------------
        quantized_z, z_indices, z_distances = self.quantize_to_subclusters(embeddings)

        # --------------------------------------------------------
        # 4. Quantize Z → clusters (Q)
        # --------------------------------------------------------
        quantized_q, q_indices, q_distances = self.quantize_to_clusters(quantized_z)

        # --------------------------------------------------------
        # 5. Decode from cluster-level representation
        # --------------------------------------------------------
        reconstructed = self.decoder(quantized_q)  # [B, T, action_dim]

        # --------------------------------------------------------
        # 6. Update codebooks if training
        # --------------------------------------------------------
        if training:
            self.update_codebook_z(embeddings, z_indices)
            self.update_codebook_q(quantized_z, q_indices)
            self.replace_dead_codes(embeddings, quantized_z)

        # --------------------------------------------------------
        # 7. ALWAYS return the full dict
        # --------------------------------------------------------
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
