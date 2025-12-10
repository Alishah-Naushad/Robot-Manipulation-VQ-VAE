import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import faiss

# ----------------------------
# LFQ Quantizer WITH EMA
# ----------------------------
def normalization(Wi, softplus_ci):  # L-inf norm
    absrowsum = torch.sum(torch.abs(Wi), dim=1, keepdim=True)
    scale = torch.minimum(
        torch.tensor(1.0, device=Wi.device),
        F.softplus(softplus_ci).unsqueeze(1) / absrowsum,
    )
    return Wi * scale


class LFQQuantizerEMA_KMeans(nn.Module):
    """
    Optimized LFQ quantizer with:
      • K-Means initialization
      • EMA codebook updates
      • Codebook usage tracking
      • Dead-code replacement
      • GPU-accelerated L2 distance computation
    """

    def __init__(
        self,
        num_codes,
        code_dim,
        decay=0.99,
        epsilon=1e-5,
        dead_threshold=5,
        replace_strategy="nearest",
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.epsilon = epsilon
        self.dead_threshold = dead_threshold
        self.replace_strategy = replace_strategy
        self.initialized = False

        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim))
        nn.init.kaiming_normal_(self.codebook)

        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_codebook", torch.randn(num_codes, code_dim))
        self.register_buffer("usage_counts", torch.zeros(num_codes))
        self.register_buffer("usage_ma", torch.zeros(num_codes))
        self.register_buffer("entropy_ma", torch.tensor(0.0))

        # Cache codebook norms for faster distance computation
        self.register_buffer("_cb_norm_cache", None)

    def kmeans_init(self, z_e):
        B, D = z_e.shape
        n_samples = min(20000, B)
        sample_idx = torch.randperm(B)[:n_samples]
        sample = z_e[sample_idx].detach().cpu().numpy()

        kmeans = KMeans(n_clusters=self.num_codes, n_init="auto", max_iter=50)
        centers = kmeans.fit(sample).cluster_centers_
        centers = torch.tensor(centers, dtype=z_e.dtype, device=z_e.device)

        self.codebook.data.copy_(centers)
        self.ema_codebook.data.copy_(centers.clone())
        self._cb_norm_cache = None  # Reset cache
        self.initialized = True

    def _compute_distances(self, z_e):
        """
        Efficient L2 distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        """
        z_e_norm = (z_e * z_e).sum(dim=1, keepdim=True)  # [B, 1]

        # Cache codebook norm for repeated calls
        if self._cb_norm_cache is None:
            self._cb_norm_cache = (self.codebook * self.codebook).sum(
                dim=1, keepdim=True
            )
        cb_norm = self._cb_norm_cache

        dots = torch.mm(z_e, self.codebook.T)  # [B, num_codes]
        distances = z_e_norm + cb_norm.T - 2.0 * dots
        return distances

    def forward(self, z_e):
        B, D = z_e.shape

        if self.training and not self.initialized:
            self.kmeans_init(z_e)

        # Compute distances and find nearest codes
        distances = self._compute_distances(z_e)
        indices = distances.argmin(dim=1)
        z_q = self.codebook[indices]

        # ---------------------------------------------------------
        # EMA updates (TRAINING ONLY)
        # ---------------------------------------------------------
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.num_codes).float()
                cluster_size = one_hot.sum(0)  # [num_codes]

                # Update EMA buffers (in-place operations for speed)
                self.ema_cluster_size.mul_(self.decay).add_(
                    cluster_size, alpha=1 - self.decay
                )
                embed_sum = one_hot.T @ z_e  # [num_codes, D]
                self.ema_codebook.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                # Update codebook
                n = self.ema_cluster_size.sum()
                cluster_size_norm = (self.ema_cluster_size + self.epsilon) / (
                    n + self.num_codes * self.epsilon
                )
                new_codebook = self.ema_codebook / cluster_size_norm.unsqueeze(1)
                self.codebook.data.copy_(new_codebook)
                self._cb_norm_cache = None  # Invalidate cache

                # Utilization tracking
                self.usage_counts.add_(cluster_size)
                self.usage_ma.mul_(0.99).add_((cluster_size > 0).float(), alpha=0.01)

                p = cluster_size / (cluster_size.sum() + 1e-8)
                entropy = -(p * torch.log(p + 1e-8)).sum()
                self.entropy_ma.mul_(0.99).add_(entropy, alpha=0.01)

                # Dead code replacement
                dead = self.usage_counts < self.dead_threshold
                if dead.any():
                    dead_idx = dead.nonzero(as_tuple=True)[0]
                    if self.replace_strategy == "nearest":
                        alive = (~dead).nonzero(as_tuple=True)[0]
                        if len(alive) > 0:
                            alive_codes = self.codebook[alive]
                            dead_codes = self.codebook[dead_idx]

                            # Vectorized distance computation
                            dead_norm = (dead_codes * dead_codes).sum(
                                dim=1, keepdim=True
                            )
                            alive_norm = (
                                (alive_codes * alive_codes).sum(dim=1, keepdim=True).T
                            )
                            dists = (
                                dead_norm
                                + alive_norm
                                - 2.0 * (dead_codes @ alive_codes.T)
                            )
                            nearest = alive[dists.argmin(dim=1)]
                            self.codebook.data[dead_idx] = self.codebook.data[nearest]
                            self._cb_norm_cache = None
                    else:
                        rand_ids = torch.randint(
                            0, B, (dead_idx.shape[0],), device=z_e.device
                        )
                        self.codebook.data[dead_idx] = z_e[rand_ids].detach()
                        self._cb_norm_cache = None

        return z_q, indices


class LipschitzMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(out_dim, in_dim))
        self.b = torch.nn.Parameter(torch.zeros(out_dim))
        self.ci = torch.nn.Parameter(torch.ones(out_dim))

    def forward(self, x):
        W_norm = normalization(self.W, self.ci)
        return torch.sigmoid(torch.matmul(x, W_norm.T) + self.b)


class LFQQuantizer(nn.Module):
    def __init__(self, num_codes, code_dim):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim))
        nn.init.kaiming_uniform_(self.codebook)

    def forward(self, z_e):
        batch_size, latent_dim = z_e.shape
        z_e_sign = (2 * torch.sign(z_e) + 1).unsqueeze(1)
        z_e_sign = torch.clamp(z_e_sign, max=1)
        z_e_expanded = z_e.unsqueeze(1)
        codebook_expanded = self.codebook.unsqueeze(0)
        distances = torch.norm(z_e_sign * (z_e_expanded - codebook_expanded), dim=-1)
        indices = torch.argmin(distances, dim=-1)
        z_q = self.codebook[indices]
        return z_q, indices


class LLFQVAE_V4(nn.Module):
    def __init__(self, feature_dim, latent_dim, num_codes=1024, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
        )
        self.to_latent = LipschitzMLP(hidden_dim, latent_dim)
        self.quantizer = LFQQuantizer(num_codes, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
        )
        self.to_output = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        h = self.encoder(x)
        z_e = self.to_latent(h)
        z_q, indices = self.quantizer(z_e)
        z_latent = z_q.clone().detach()
        recon = self.decoder(z_q)
        x_recon = self.to_output(recon)

        recon_loss = F.mse_loss(x_recon, x)
        commitment_loss = F.mse_loss(z_q.detach(), z_e)
        codebook_loss = F.mse_loss(z_q, z_e.detach())

        loss = recon_loss + 0.25 * commitment_loss + 0.25 * codebook_loss
        return z_latent, loss


class HierarchicalLFQHVQVAE(nn.Module):
    def __init__(
        self,
        feature_dim,
        z_dim,
        q_dim,
        num_z_codes=1024,
        num_q_codes=512,
        hidden_dim=128,
    ):
        super().__init__()

        # Encoder
        self.encoder = LLFQVAE_V4(
            feature_dim=feature_dim,
            latent_dim=z_dim,
            num_codes=num_z_codes,
            hidden_dim=hidden_dim,
        ).encoder

        self.to_z_latent = LLFQVAE_V4(
            feature_dim=feature_dim,
            latent_dim=z_dim,
            num_codes=num_z_codes,
            hidden_dim=hidden_dim,
        ).to_latent

        # Quantizers
        self.z_quantizer = LFQQuantizerEMA_KMeans(num_z_codes, z_dim, dead_threshold=3)
        self.q_quantizer = LFQQuantizerEMA_KMeans(num_q_codes, q_dim, dead_threshold=1)
        self.q_encoder = LipschitzMLP(z_dim, q_dim)

        # Decoder
        self.decoder = LLFQVAE_V4(
            feature_dim=feature_dim,
            latent_dim=q_dim,
            num_codes=num_q_codes,
            hidden_dim=hidden_dim,
        ).decoder

        self.to_output = LLFQVAE_V4(
            feature_dim=feature_dim,
            latent_dim=q_dim,
            num_codes=num_q_codes,
            hidden_dim=hidden_dim,
        ).to_output

    def forward(self, x):
        # ============================
        # 1) Z-LEVEL
        # ============================
        h = self.encoder(x)
        z_e = self.to_z_latent(h)
        z_q, z_idx = self.z_quantizer(z_e)

        commit_z = F.mse_loss(z_e, z_q.detach())
        codebook_z = F.mse_loss(z_q, z_e.detach())

        # ============================
        # 2) Q-LEVEL (hierarchical)
        # ============================
        q_e = self.q_encoder(z_q.detach())
        q_q, q_idx = self.q_quantizer(q_e)

        commit_q = F.mse_loss(q_e, q_q.detach())
        codebook_q = F.mse_loss(q_q, q_e.detach())

        # ============================
        # Reconstruction
        # ============================
        dec_h = self.decoder(q_q)
        x_recon = self.to_output(dec_h)
        recon_loss = F.mse_loss(x_recon, x)

        # ============================
        # Total loss
        # ============================
        loss = (
            recon_loss + 0.25 * (commit_z + codebook_z) + 0.25 * (commit_q + codebook_q)
        )

        # Compute utilization metrics (no_grad for efficiency)
        # with torch.no_grad():
        #     z_used = (self.z_quantizer.ema_cluster_size > 0).sum().item()
        #     q_used = (self.q_quantizer.ema_cluster_size > 0).sum().item()
        #     z_util = (self.z_quantizer.ema_cluster_size > 0).float().mean().item()
        #     q_util = (self.q_quantizer.ema_cluster_size > 0).float().mean().item()
        with torch.no_grad():
            z_used = 0
            q_used = 0
            z_util = 0
            q_util = 0

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "z_commit": commit_z,
            "z_codebook": codebook_z,
            "q_commit": commit_q,
            "q_codebook": codebook_q,
            "x_recon": x_recon,
            "z_q": z_q,
            "q_q": q_q,
            "z_indices": z_idx,
            "q_indices": q_idx,
            "z_used": z_used,
            "q_used": q_used,
            "z_util": z_util,
            "q_util": q_util,
        }
