import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


# ----------------------------
# LFQ Quantizer WITH EMA
# ----------------------------
def normalization(Wi, softplus_ci):  # L-inf norm
    absrowsum = torch.sum(torch.abs(Wi), dim=1, keepdim=True)  # Shape: (out_dim, 1)
    scale = torch.minimum(
        torch.tensor(1.0, device=Wi.device),
        F.softplus(softplus_ci).unsqueeze(1) / absrowsum,
    )
    return Wi * scale  # Broadcasting should now work


class LFQQuantizerEMA_KMeans(nn.Module):
    """
    LFQ quantizer with:
      • K-Means initialization (first batch)
      • EMA codebook updates
      • Codebook usage tracking
      • Dead-code replacement
      • L-inf Lipschitz ML layer compatibility
    """

    def __init__(
        self,
        num_codes,
        code_dim,
        decay=0.99,
        epsilon=1e-5,
        dead_threshold=5,         # minimum usage before marking as dead
        replace_strategy="nearest",  # or "random"
    ):
        super().__init__()

        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.epsilon = epsilon
        self.dead_threshold = dead_threshold
        self.replace_strategy = replace_strategy

        # K-Means will overwrite codebook on first forward pass
        self.initialized = False

        # Main codebook + EMA buffers
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim))
        nn.init.kaiming_normal_(self.codebook)

        self.ema_cluster_size = nn.Parameter(torch.zeros(num_codes), requires_grad=False)
        self.ema_codebook = nn.Parameter(torch.randn(num_codes, code_dim), requires_grad=False)

        # Tracking utilization
        self.register_buffer("usage_counts", torch.zeros(num_codes))
        self.register_buffer("usage_ma", torch.zeros(num_codes))  # moving average
        self.register_buffer("entropy_ma", torch.tensor(0.0))

    # -------------------------------------------------------------
    # K-Means initialization from first batch
    # -------------------------------------------------------------
    def kmeans_init(self, z_e):
        B, D = z_e.shape
        n_samples = min(20000, B)  # cap for memory

        sample_idx = torch.randperm(B)[:n_samples]
        sample = z_e[sample_idx].detach().cpu().numpy()

        kmeans = KMeans(n_clusters=self.num_codes, n_init="auto", max_iter=50)
        centers = kmeans.fit(sample).cluster_centers_
        centers = torch.tensor(centers, dtype=z_e.dtype, device=z_e.device)

        self.codebook.data.copy_(centers)
        self.ema_codebook.data.copy_(centers.clone())
        self.initialized = True

    # -------------------------------------------------------------
    # Forward quantization (LFQ + codebook lookup)
    # -------------------------------------------------------------
    def forward(self, z_e):
        B, D = z_e.shape

        # ---- Run KMEANS on first forward ----
        if not self.initialized:
            self.kmeans_init(z_e)
            # After init, continue as usual

        # ---- Lipschitz sign-flip rule ----
        z_e_sign = (2 * torch.sign(z_e) + 1).clamp(max=1).unsqueeze(1)
        z_e_expanded = z_e.unsqueeze(1)
        cb_expanded = self.codebook.unsqueeze(0)

        distances = torch.norm(z_e_sign * (z_e_expanded - cb_expanded), dim=-1)
        indices = torch.argmin(distances, dim=-1)
        z_q = self.codebook[indices]

        # ---------------------------------------------------------
        # Update EMA codebook (no grad)
        # ---------------------------------------------------------
        with torch.no_grad():
            # one-hot cluster counts
            one_hot = F.one_hot(indices, self.num_codes).float()
            cluster_size = one_hot.sum(0)

            # EMA
            self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

            embed_sum = one_hot.T @ z_e  # [K, D]
            self.ema_codebook.mul_(self.decay).add_(embed_sum, alpha=(1 - self.decay))

            # Normalize
            n = self.ema_cluster_size.sum()
            cluster_size_norm = (self.ema_cluster_size + self.epsilon) / (n + self.num_codes * self.epsilon)
            new_codebook = self.ema_codebook / cluster_size_norm.unsqueeze(1)
            self.codebook.data.copy_(new_codebook)

        # ---------------------------------------------------------
        # Track utilization statistics
        # ---------------------------------------------------------
        with torch.no_grad():
            self.usage_counts += cluster_size
            self.usage_ma = 0.99 * self.usage_ma + 0.01 * (cluster_size > 0).float()

            # entropy of usage distribution
            p = cluster_size / (cluster_size.sum() + 1e-8)
            entropy = -(p * (p + 1e-8).log()).sum()
            self.entropy_ma = 0.99 * self.entropy_ma + 0.01 * entropy

        # ---------------------------------------------------------
        # Dead code replacement
        # ---------------------------------------------------------
        dead = self.usage_counts < self.dead_threshold
        if dead.any():
            dead_idx = dead.nonzero(as_tuple=False).squeeze(1)

            if self.replace_strategy == "nearest":
                # Replace with nearest active code
                alive = (~dead).nonzero(as_tuple=False).squeeze(1)
                if len(alive) > 0:
                    alive_codes = self.codebook[alive]
                    for idx in dead_idx:
                        code = self.codebook[idx]
                        dist = torch.norm(alive_codes - code.unsqueeze(0), dim=-1)
                        nearest = alive[torch.argmin(dist)]
                        self.codebook.data[idx] = self.codebook.data[nearest]
            else:
                # Random data sample
                rand_ids = torch.randint(0, B, (len(dead_idx),), device=z_e.device)
                self.codebook.data[dead_idx] = z_e[rand_ids].detach()

        return z_q, indices
class LipschitzMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(out_dim, in_dim))
        self.b = torch.nn.Parameter(torch.zeros(out_dim))
        self.ci = torch.nn.Parameter(torch.ones(out_dim))  # Learnable ci parameter

    def forward(self, x):
        W_norm = normalization(self.W, self.ci)
        return torch.sigmoid(torch.matmul(x, W_norm.T) + self.b)

class LFQQuantizer(nn.Module):
    def __init__(self, num_codes, code_dim):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.codebook = nn.Parameter(
            torch.randn(num_codes, code_dim)
        )  # Learnable codebook
        nn.init.kaiming_uniform_(self.codebook)  # Proper initialization

    def forward(self, z_e):
        batch_size, latent_dim = z_e.shape  # Ensure shape consistency
        z_e_sign = (2 * torch.sign(z_e) + 1).unsqueeze(1)  # Shape: [B, 1, latent_dim]
        z_e_sign = torch.clamp(z_e_sign, max=1)
        z_e_expanded = z_e.unsqueeze(1)  # Shape: [B, 1, D]
        codebook_expanded = self.codebook.unsqueeze(0)  # Shape: [1, num_codes, D]
        distances = torch.norm(
            z_e_sign * (z_e_expanded - codebook_expanded), dim=-1
        )  # Compute L2 distances
        indices = torch.argmin(distances, dim=-1)  # Get closest code
        z_q = self.codebook[indices]  # Retrieve quantized values
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
        self.quantizer = LFQQuantizer(num_codes,latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
        )
        self.to_output = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        h = self.encoder(x)  # Correct shape: [B, hidden_dim]
        z_e = self.to_latent(h)  # Shape: [B, latent_dim]
        z_q, indices = self.quantizer(z_e)  # Shape: [B, latent_dim]
        z_latent = z_q.clone().detach()
        recon = self.decoder(z_q)  # Correct shape: [B, hidden_dim]
        x_recon = self.to_output(recon)  # Shape: [B, feature_dim]

        # Compute losses
        recon_loss = F.mse_loss(x_recon, x)  # Reconstruction loss
        commitment_loss = F.mse_loss(z_q.detach(), z_e)  # Commitment loss
        codebook_loss = F.mse_loss(z_q, z_e.detach())  # Codebook loss

        loss = recon_loss + 0.25 * commitment_loss + 0.25 * codebook_loss
        return z_latent, loss

class LFQQuantizerEMA(nn.Module):
    def __init__(self, num_codes, code_dim, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim

        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim))
        nn.init.kaiming_uniform_(self.codebook)

        # EMA buffers
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_codebook", torch.randn(num_codes, code_dim))

        self.decay = decay
        self.epsilon = epsilon

    def forward(self, z_e):
        """
        z_e: (B, D)
        Returns: z_q, indices
        """

        # LFQ sign logic
        z_e_sign = (2 * torch.sign(z_e) + 1).unsqueeze(1)
        z_e_sign = torch.clamp(z_e_sign, max=1)

        # distances
        z_e_exp = z_e.unsqueeze(1)
        cb_exp = self.codebook.unsqueeze(0)

        distances = torch.norm(z_e_sign * (z_e_exp - cb_exp), dim=-1)
        indices = torch.argmin(distances, dim=-1)
        z_q = self.codebook[indices]

        # -----------------------
        # EMA UPDATE
        # -----------------------

        if self.training:
            # one-hot encode selected codes
            encodings = F.one_hot(indices, self.num_codes).type(z_e.dtype)

            # update cluster size
            self.ema_cluster_size.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )

            # update codebook averages
            embed_sum = encodings.t() @ z_e
            self.ema_codebook.mul_(self.decay).add_(
                embed_sum, alpha=1 - self.decay
            )

            # normalize
            n = self.ema_cluster_size + self.epsilon
            self.codebook.data.copy_(self.ema_codebook / n.unsqueeze(1))

        return z_q, indices



# ============================================================
# NEW HIERARCHICAL HVQ-VAE USING EITHER Z-LEVEL AND Q-LEVEL LFQ
# ============================================================

class HierarchicalLFQHVQVAE(nn.Module):
    def __init__(
        self,
        feature_dim,
        z_dim,
        q_dim,
        num_z_codes=1024,
        num_q_codes=512,
        hidden_dim=128
    ):
        super().__init__()

        # -------------------------------
        # Encoder (shared LLFQVAE_V4)
        # -------------------------------
        self.encoder = LLFQVAE_V4(
            feature_dim=feature_dim,
            latent_dim=z_dim,
            num_codes=num_z_codes,
            hidden_dim=hidden_dim,
        ).encoder  # USE ONLY encoder part

        self.to_z_latent = LLFQVAE_V4(
            feature_dim=feature_dim,
            latent_dim=z_dim,
            num_codes=num_z_codes,
            hidden_dim=hidden_dim,
        ).to_latent  # Lipschitz mapping

        # -------------------------------
        # Quantizers (Z then Q)
        # -------------------------------
        #self.z_quantizer = LFQQuantizerEMA(num_z_codes, z_dim)

        self.z_quantizer = LFQQuantizerEMA_KMeans(num_z_codes, z_dim,dead_threshold=3)
        self.q_quantizer = LFQQuantizerEMA_KMeans(num_q_codes, q_dim,dead_threshold=1)
        self.q_encoder = LipschitzMLP(z_dim, q_dim)
        #self.q_quantizer = LFQQuantizerEMA(num_q_codes, q_dim)

        # -------------------------------
        # Decoder (shared LLFQVAE_V4)
        # -------------------------------
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

    # ------------------------------------------------------------
    # Helper for losses
    # ------------------------------------------------------------
    def vq_loss(self, z_e, z_q):
        """
        Standard VQ losses adapted to LFQ (detached STE).
        """
        commit = F.mse_loss(z_e, z_q.detach())
        codebook = F.mse_loss(z_q, z_e.detach())
        return commit, codebook

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x):

        # ============================
        # 1) Z-LEVEL
        # ============================
        h = self.encoder(x)
       
        z_e = self.to_z_latent(h)   
                       # (B, z_dim)
        z_q, z_idx = self.z_quantizer(z_e)            # EMA quantization
        #commit_z, codebook_z = self.vq_loss(z_e, z_q)
        commit_z= F.mse_loss(z_e, z_q.detach())
        codebook_z = F.mse_loss(z_q, z_e.detach())
        # ============================
        # 2) Q-LEVEL (hierarchical)
        # ============================
        q_e = self.q_encoder(z_q.detach())                     # (B, q_dim)
        q_q, q_idx = self.q_quantizer(q_e)
        #commit_q, codebook_q = self.vq_loss(q_e, q_q)
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
            recon_loss
            + 0.25 * (commit_z + codebook_z)
            + 0.25 * (commit_q + codebook_q)
        )
        with torch.no_grad(): 
            z_used = (self.z_quantizer.ema_cluster_size > 0).sum().item()
            q_used = (self.q_quantizer.ema_cluster_size > 0).sum().item() 
            z_util = (self.z_quantizer.ema_cluster_size > 0).float().mean().item()
            q_util = (self.q_quantizer.ema_cluster_size > 0).float().mean().item()

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
        }