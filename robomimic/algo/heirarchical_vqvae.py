import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


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


def laplace_smoothing(cluster_size, num_embeddings, eps=1e-6):
    # Avoid zeros and extremely small denominators for EMA normalization
    smoothed = cluster_size.clone()
    smoothed = smoothed + eps
    return smoothed


class HierarchicalVQVAE(nn.Module):
    """
    HVQ-VAE aligned with Double-HVQ methodology:
      - Two-stage quantization: embeddings -> Z (subclusters) -> Q (clusters)
      - STE used at BOTH levels (encoder receives gradient)
      - EMA updates for prototypes
      - L2 normalization + cosine similarity quantization
      - KMeans initialization option (on current batch)
      - Safe dead-code replacement using normalized batch samples
    """

    def __init__(
        self,  # ActionDecoder instance (user-supplied)
        action_dim: int,
        embed_dim: int = 512,
        num_subclusters: int = 128,  # |Z|
        num_clusters: int = 64,  # |Q|
        commitment_cost: float = 1.0,
        ema_decay: float = 0.99,
        num_stages=2,
        beta=0.8,
        num_layers_per_stage=10,
        dropout=0.1,
        eps: float = 1e-5,
        replace_threshold_z: float = 1.0,
        replace_threshold_q: float = 1.0,
        kmeans_init: bool = True,
        use_cosine: bool = True,
        normalize_after_ema: bool = True,
        verbose: bool = False,
    ):
        super().__init__()

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

        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.num_subclusters = num_subclusters
        self.num_clusters = num_clusters

        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.eps = eps
        self.replace_threshold_z = replace_threshold_z
        self.replace_threshold_q = replace_threshold_q
        self.kmeans_init = kmeans_init
        self.use_cosine = use_cosine
        self.normalize_after_ema = normalize_after_ema
        self.verbose = verbose

        # codebooks: shape [K, D]
        self.register_buffer(
            "codebook_z", torch.randn(self.num_subclusters, self.embed_dim)
        )
        self.register_buffer(
            "codebook_q", torch.randn(self.num_clusters, self.embed_dim)
        )

        # initialize small uniform then normalize
        nn.init.uniform_(
            self.codebook_z, -1.0 / self.num_subclusters, 1.0 / self.num_subclusters
        )
        nn.init.uniform_(
            self.codebook_q, -1.0 / self.num_clusters, 1.0 / self.num_clusters
        )

        # EMA statistics (buffers, not parameters)
        # cluster_size tracks EMA of counts (shape [K])
        self.register_buffer("cluster_size_z", torch.zeros(self.num_subclusters))
        self.register_buffer("embed_avg_z", self.codebook_z.clone())

        self.register_buffer("cluster_size_q", torch.zeros(self.num_clusters))
        self.register_buffer("embed_avg_q", self.codebook_q.clone())

        # whether codebooks were initialized by kmeans
        self.register_buffer(
            "codebooks_initialized", torch.tensor(False, dtype=torch.bool)
        )

        # ensure initial normalization
        if self.use_cosine:
            self.codebook_z.copy_(F.normalize(self.codebook_z, dim=-1))
            self.codebook_q.copy_(F.normalize(self.codebook_q, dim=-1))

    @torch.no_grad()
    def initialize_codebooks_with_current_batch(self, embeddings, sample_limit=10000):
        """
        Initialize codebooks with a KMeans on the current batch embeddings (optional).
        embeddings: [B, T, D] (assumed not yet normalized)
        """
        if self.codebooks_initialized:
            return

        device = embeddings.device
        B, T, D = embeddings.shape
        flat = embeddings.reshape(-1, D).cpu().numpy()

        if flat.shape[0] > sample_limit:
            idx = np.random.choice(flat.shape[0], sample_limit, replace=False)
            sample = flat[idx]
        else:
            sample = flat

        if self.kmeans_init:
            # Z init
            km_z = KMeans(
                n_clusters=self.num_subclusters,
                n_init=10,
                random_state=42,
                max_iter=300,
            )
            km_z.fit(sample)
            centers_z = torch.from_numpy(km_z.cluster_centers_).float().to(device)

            # Q init (we run KMeans to a smaller k for Q)
            km_q = KMeans(
                n_clusters=self.num_clusters, n_init=10, random_state=42, max_iter=300
            )
            km_q.fit(sample)
            centers_q = torch.from_numpy(km_q.cluster_centers_).float().to(device)
            print("*" * 25)
            print("kmeans initialization complete")
            print("*" * 25)
        else:
            centers_z = torch.randn(self.num_subclusters, self.embed_dim, device=device)
            centers_q = torch.randn(self.num_clusters, self.embed_dim, device=device)

        with torch.no_grad():
            if self.use_cosine:
                centers_z = F.normalize(centers_z, dim=-1)
                centers_q = F.normalize(centers_q, dim=-1)
            self.codebook_z.copy_(centers_z)
            self.codebook_q.copy_(centers_q)

            # initialize EMA statistics from current batch assignment counts
            # simple pass to compute counts
            flat_t = (
                torch.from_numpy(sample).to(device)
                if isinstance(sample, np.ndarray)
                else torch.from_numpy(sample).to(device)
            )
            flat_t = flat_t.view(-1, D).to(device)
            if self.use_cosine:
                flat_norm = F.normalize(flat_t, dim=-1)
                sim_z = torch.matmul(flat_norm, self.codebook_z.t())  # [N, |Z|]
                assign_z = torch.argmax(sim_z, dim=1)
            else:
                dists = torch.cdist(flat_t, self.codebook_z)  # [N, |Z|]
                assign_z = torch.argmin(dists, dim=1)

            counts_z = torch.bincount(assign_z, minlength=self.num_subclusters).float()
            self.cluster_size_z.copy_(counts_z)
            # embed_avg_z = sum assigned embeddings
            sums_z = torch.zeros_like(self.embed_avg_z)
            for k in range(self.num_subclusters):
                mask = (assign_z == k).nonzero(as_tuple=False).squeeze(-1)
                if mask.numel() > 0:
                    sums_z[k] = flat_t[mask].sum(dim=0)
            self.embed_avg_z.copy_(sums_z)

            # same for Q
            if self.use_cosine:
                sim_q = torch.matmul(flat_norm, self.codebook_q.t())
                assign_q = torch.argmax(sim_q, dim=1)
            else:
                dists_q = torch.cdist(flat_t, self.codebook_q)
                assign_q = torch.argmin(dists_q, dim=1)

            counts_q = torch.bincount(assign_q, minlength=self.num_clusters).float()
            self.cluster_size_q.copy_(counts_q)
            sums_q = torch.zeros_like(self.embed_avg_q)
            for k in range(self.num_clusters):
                mask = (assign_q == k).nonzero(as_tuple=False).squeeze(-1)
                if mask.numel() > 0:
                    sums_q[k] = flat_t[mask].sum(dim=0)
            self.embed_avg_q.copy_(sums_q)

            self.codebooks_initialized.fill_(True)

        if self.verbose:
            print(
                f"[KMeans init] Z centers: {self.num_subclusters}, Q centers: {self.num_clusters}"
            )

    def _compute_similarities(self, vectors, codebook):
        """
        vectors: [N, D] (assumed normalized if using cosine)
        codebook: [K, D] (assumed normalized if using cosine)
        returns similarities [N, K] (cosine if normalized else negative L2)
        """
        if self.use_cosine:
            # assume both are normalized
            return torch.matmul(vectors, codebook.t())  # higher = better
        else:
            # return negative distances to use argmax
            return -torch.cdist(vectors, codebook)  # higher = better (less distance)

    def quantize_z(self, embeddings, use_ste=True):
        """
        embeddings: [B, T, D] (not necessarily normalized)
        Returns: quantized_z [B, T, D], z_indices [B, T], z_distances [B, T, |Z|]
        """
        B, T, D = embeddings.shape
        flat = embeddings.reshape(-1, D)
        if self.use_cosine:
            flat_norm = F.normalize(flat, dim=-1)
            code_z = F.normalize(self.codebook_z, dim=-1)
        else:
            flat_norm = flat
            code_z = self.codebook_z

        sims = self._compute_similarities(flat_norm, code_z)  # [N, |Z|]
        idx_flat = torch.argmax(sims, dim=1)  # [N]
        quantized_flat = self.codebook_z[idx_flat]  # [N, D]
        quantized = quantized_flat.view(B, T, D)

        # STE: encoder receives gradient if use_ste True
        if use_ste:
            if self.use_cosine:
                embeddings_norm = F.normalize(embeddings, dim=-1)
                quantized = embeddings_norm + (quantized - embeddings_norm).detach()
            else:
                quantized = embeddings + (quantized - embeddings).detach()

        distances = (1.0 - sims).view(
            B, T, -1
        )  # if cosine: 1 - sim; if negative L2, behaves similarly
        indices = idx_flat.view(B, T)
        return quantized, indices, distances

    def quantize_q(self, subclusters, use_ste=True):
        """
        subclusters: [B, T, D] (quantized z output)
        Returns: quantized_q [B, T, D], q_indices [B, T], q_distances [B, T, |Q|]
        """
        B, T, D = subclusters.shape
        flat = subclusters.reshape(-1, D)
        if self.use_cosine:
            flat_norm = F.normalize(flat, dim=-1)
            code_q = F.normalize(self.codebook_q, dim=-1)
        else:
            flat_norm = flat
            code_q = self.codebook_q

        sims = self._compute_similarities(flat_norm, code_q)
        idx_flat = torch.argmax(sims, dim=1)
        quantized_flat = self.codebook_q[idx_flat]
        quantized = quantized_flat.view(B, T, D)

        # STE (important)
        if use_ste:
            if self.use_cosine:
                sub_norm = F.normalize(subclusters, dim=-1)
                quantized = sub_norm + (quantized - sub_norm).detach()
            else:
                quantized = subclusters + (quantized - subclusters).detach()

        distances = (1.0 - sims).view(B, T, -1)
        indices = idx_flat.view(B, T)
        return quantized, indices, distances

    @torch.no_grad()
    def _ema_update(
        self,
        embeddings_flat,
        indices_flat,
        embed_avg_buffer,
        cluster_size_buffer,
        codebook_buffer,
    ):
        """
        Generic EMA updater for a codebook.
        embeddings_flat: [N, D]
        indices_flat: [N] indexes into codebook
        embed_avg_buffer: buffer to update (shape [K, D])
        cluster_size_buffer: buffer to update (shape [K])
        codebook_buffer: buffer to write final prototypes (shape [K, D]) -- updated inside
        """
        K = cluster_size_buffer.numel()
        device = embeddings_flat.device

        enc_onehot = F.one_hot(indices_flat, num_classes=K).type_as(
            embeddings_flat
        )  # [N, K]
        counts = enc_onehot.sum(dim=0)  # [K]

        # EMA update of counts and sums
        cluster_size_buffer.mul_(self.ema_decay).add_(
            counts, alpha=(1.0 - self.ema_decay)
        )
        embed_sum = enc_onehot.t() @ embeddings_flat  # [K, D]
        embed_avg_buffer.mul_(self.ema_decay).add_(
            embed_sum, alpha=(1.0 - self.ema_decay)
        )

        # compute normalized prototypes safely
        denom = laplace_smoothing(cluster_size_buffer, K, eps=self.eps).unsqueeze(
            1
        )  # [K, 1]
        new_prototypes = embed_avg_buffer / denom

        if self.normalize_after_ema and self.use_cosine:
            new_prototypes = F.normalize(new_prototypes, dim=-1)

        codebook_buffer.copy_(new_prototypes)

    @torch.no_grad()
    def replace_dead_codes(self, embeddings, subclusters):
        """
        Replace dead prototypes with normalized samples from the current batch.
        embeddings: [B, T, D] raw encoder embeddings (not necessarily normalized)
        subclusters: [B, T, D] quantized z outputs (already quantized)
        """
        # Z
        dead_mask_z = self.cluster_size_z < self.replace_threshold_z
        if dead_mask_z.any():
            num_dead = int(dead_mask_z.sum().item())
            B, T, D = embeddings.shape
            emb_flat = embeddings.reshape(-1, D)
            # sample normalized vectors
            if emb_flat.size(0) == 0:
                return
            idx = torch.randint(
                0, emb_flat.size(0), (num_dead,), device=emb_flat.device
            )
            repl = emb_flat[idx]  # raw
            if self.use_cosine:
                repl = F.normalize(repl, dim=-1)
            with torch.no_grad():
                # replace prototypes and reset stats for those entries
                self.codebook_z[dead_mask_z] = repl
                self.embed_avg_z[dead_mask_z] = repl
                self.cluster_size_z[dead_mask_z] = 1.0

        # Q
        dead_mask_q = self.cluster_size_q < self.replace_threshold_q
        if dead_mask_q.any():
            num_dead_q = int(dead_mask_q.sum().item())
            B, T, D = subclusters.shape
            sub_flat = subclusters.reshape(-1, D)
            if sub_flat.size(0) == 0:
                return
            idxq = torch.randint(
                0, sub_flat.size(0), (num_dead_q,), device=sub_flat.device
            )
            repl_q = sub_flat[idxq]
            if self.use_cosine:
                repl_q = F.normalize(repl_q, dim=-1)
            with torch.no_grad():
                self.codebook_q[dead_mask_q] = repl_q
                self.embed_avg_q[dead_mask_q] = repl_q
                self.cluster_size_q[dead_mask_q] = 1.0

    def forward(self, actions, training=True):
        """
        actions: [B, T, action_dim]
        returns dict with embeddings, quantized_z, quantized_q, recon, indices, distances
        """
        # 1) encode
        embeddings = self.encoder(actions)  # [B, T, D] -> should be float32
        # keep a normalized copy for quantization but keep embeddings for commit loss (we normalize inside losses)
        if not self.codebooks_initialized:
            # init from current batch (kmeans optional)
            self.initialize_codebooks_with_current_batch(embeddings)

        # 2) quantize z (use STE)
        quantized_z, z_indices, z_dists = self.quantize_z(embeddings, use_ste=True)

        # 3) quantize q (use STE)
        quantized_q, q_indices, q_dists = self.quantize_q(quantized_z, use_ste=True)

        # 4) decode from quantized_q
        reconstructed = self.decoder(quantized_q)

        # 5) update codebooks & replace dead codes (EMA) during training
        if training:
            B, T, D = embeddings.shape
            emb_flat = embeddings.reshape(-1, D)
            z_idx_flat = z_indices.reshape(-1)
            self._ema_update(
                emb_flat,
                z_idx_flat,
                self.embed_avg_z,
                self.cluster_size_z,
                self.codebook_z,
            )

            sub_flat = quantized_z.reshape(-1, D)
            q_idx_flat = q_indices.reshape(-1)
            self._ema_update(
                sub_flat,
                q_idx_flat,
                self.embed_avg_q,
                self.cluster_size_q,
                self.codebook_q,
            )

            # replace dead prototypes from batch samples (normalized)
            self.replace_dead_codes(embeddings, quantized_z)

        return {
            "embeddings": embeddings,
            "quantized_z": quantized_z,
            "quantized_q": quantized_q,
            "reconstructed_actions": reconstructed,
            "z_indices": z_indices,
            "q_indices": q_indices,
            "z_distances": z_dists,
            "q_distances": q_dists,
        }

    # --- losses and helpers ---
    def compute_reconstruction_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def compute_commitment_loss_z(self, embeddings, quantized_z):
        # use normalized MSE consistent with cosine quantization
        if self.use_cosine:
            e_norm = F.normalize(embeddings, dim=-1)
            qz_norm = F.normalize(quantized_z.detach(), dim=-1)
            return F.mse_loss(e_norm, qz_norm)
        else:
            return F.mse_loss(embeddings, quantized_z.detach())

    def compute_commitment_loss_q(self, quantized_z, quantized_q):
        if self.use_cosine:
            z_norm = F.normalize(quantized_z, dim=-1)
            q_norm = F.normalize(quantized_q.detach(), dim=-1)
            return F.mse_loss(z_norm, q_norm)
        else:
            return F.mse_loss(quantized_z, quantized_q.detach())

    def compute_vqvae_loss(self, outputs, true_actions, lambda_rec=1.0):
        L_rec = self.compute_reconstruction_loss(
            outputs["reconstructed_actions"], true_actions
        )
        L_cz = self.compute_commitment_loss_z(
            outputs["embeddings"], outputs["quantized_z"]
        )
        L_cq = self.compute_commitment_loss_q(
            outputs["quantized_z"], outputs["quantized_q"]
        )
        total = self.commitment_cost * (L_cz + L_cq) + lambda_rec * L_rec
        return {
            "vqvae_loss": total,
            "L_rec": L_rec,
            "L_commit_z": L_cz,
            "L_commit_q": L_cq,
        }

    def get_codebook_usage(self):
        with torch.no_grad():
            z_used = (self.cluster_size_z > 0).sum().item()
            q_used = (self.cluster_size_q > 0).sum().item()
            return {
                "z_used": z_used,
                "z_total": self.num_subclusters,
                "z_usage_pct": 100.0 * z_used / max(1, self.num_subclusters),
                "z_utilization": (self.cluster_size_z > 0).float().mean().item(),
                "q_utilization": (self.cluster_size_q > 0).float().mean().item(),
                "q_used": q_used,
                "q_total": self.num_clusters,
                "q_usage_pct": 100.0 * q_used / max(1, self.num_clusters),
                "z_dead": int(
                    (self.cluster_size_z < self.replace_threshold_z).sum().item()
                ),
                "q_dead": int(
                    (self.cluster_size_q < self.replace_threshold_q).sum().item()
                ),
            }
