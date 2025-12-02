# Hierarchical VQ-VAE Implementation Plan for ICL Transformer

## Overview
This document outlines the step-by-step plan to integrate hierarchical two-stage VQ-VAE (coarse to fine) into the **ICLTransformer** class (non-GMM) for robot action tokenization and prediction.

## Current State Analysis

### Existing ICL Code Structure
- **Base Class**: `ICLTransformer` (extends `ICL`)
- **Current State**: Direct continuous action prediction from transformer
- **Input**: Robot observations (low-dim states: EEF pos, quat, gripper, object)
- **Output**: Continuous action predictions `[B, T, action_dim]`
- **Context**: Uses observation sequences with context length for in-context learning
- **Supervision**: Can supervise all timesteps or just final timestep

### Target Hierarchical VQ-VAE Methodology
- **Two-level quantization**: Actions → Subclusters (Z) → Clusters (Q)
- **Architecture**: MSTCN-based encoder/decoder with dilated temporal convolutions
- **Training**: Reconstruction + two commitment losses
- **Inference**: Soft assignment via FIFA with cosine similarity
- **Goal**: Tokenize actions hierarchically, then predict continuous actions from tokenized context

---

## Implementation Plan

### Phase 1: Architecture Design & Preparation

#### Step 1.1: Define Hierarchical VQ-VAE Network Architecture
**Goal**: Create the hierarchical VQ-VAE model adapted for robot actions

**Components to Implement**:
1. **Action Encoder** (replacing frame encoder)
   - Input: Action sequences `[batch, seq_len, action_dim]`
   - Architecture: Light MSTCN with 2 stages, 10 layers each
   - Dilated temporal convolutions (dilation doubles per layer)
   - Output: Encoded embeddings `e_mt` for each timestep

2. **Action Decoder** (replacing frame decoder)
   - Input: Quantized vectors `q_mt`
   - Architecture: Mirror of encoder
   - Output: Reconstructed actions `[batch, seq_len, action_dim]`

3. **Two-Level Codebook System**
   - **Subcluster Codebook Z**: `[num_subclusters, embed_dim]`
   - **Cluster Codebook Q**: `[num_clusters, embed_dim]`
   - L2 normalization on embeddings and codebook vectors

**Key Decisions**:
- `num_subclusters` (|Z|): 128-256 (configurable, smaller for robot actions)
- `num_clusters` (|Q|): 32-64 (configurable)
- `embed_dim`: 512 (matching transformer embed_dim)
- `action_dim`: 7-10 (robot action dimension)
- Dilation pattern: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] per stage

#### Step 1.2: Create Network Module Class
**File**: Create new file `robomimic/models/hierarchical_vqvae_nets.py`

**Classes to Implement**:
```python
class DilatedTemporalConvLayer(nn.Module):
    """Single dilated temporal convolution layer"""
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3):
        # 1D temporal convolution with dilation
        # Input: [B, C, T]
        # Output: [B, C, T]
    
class MSTCNStage(nn.Module):
    """Single stage with 10 dilated conv layers"""
    def __init__(self, num_layers, in_channels, out_channels):
        # Stack of 10 layers with increasing dilation
        # Dilation: 2^0, 2^1, 2^2, ..., 2^9
    
class ActionEncoder(nn.Module):
    """MSTCN-based action encoder"""
    def __init__(self, action_dim, embed_dim, num_stages=2):
        # Initial projection: action_dim -> embed_dim
        # 2 MSTCN stages
        # Output: embeddings [B, T, D]
    
class ActionDecoder(nn.Module):
    """MSTCN-based action decoder"""
    def __init__(self, embed_dim, action_dim, num_stages=2):
        # 2 MSTCN stages
        # Final projection: embed_dim -> action_dim
        # Output: reconstructed actions [B, T, action_dim]
    
class HierarchicalVQVAE(nn.Module):
    """Complete hierarchical VQ-VAE with two-level quantization"""
    def __init__(self, action_dim, num_subclusters, num_clusters, 
                 embed_dim, num_stages=2, beta=0.8):
        # Encoder, Decoder, Codebooks Z and Q
        # EMA tracking variables
```

---

### Phase 2: Quantization & Codebook Management

#### Step 2.1: Implement Vector Quantization Logic
**Location**: Within `HierarchicalVQVAE` class

**Functions to Implement**:

1. **`quantize_to_subclusters(embeddings)`**
   ```python
   def quantize_to_subclusters(self, embeddings):
       """
       Args:
           embeddings: [B, T, D] encoder output
       Returns:
           quantized: [B, T, D] quantized vectors from codebook Z
           indices: [B, T] indices into codebook Z
           distances: [B, T, |Z|] cosine distances
       """
       # L2 normalize embeddings
       embeddings = F.normalize(embeddings, dim=-1)
       codebook_z = F.normalize(self.codebook_z, dim=-1)
       
       # Compute cosine similarity: sim = (z_j · e_mt) / (||z_j|| ||e_mt||)
       similarities = torch.matmul(embeddings, codebook_z.t())  # [B, T, |Z|]
       
       # Assign to nearest (max similarity)
       indices = torch.argmax(similarities, dim=-1)  # [B, T]
       
       # Lookup quantized vectors
       quantized = self.codebook_z[indices]  # [B, T, D]
       
       # Straight-through estimator
       quantized = embeddings + (quantized - embeddings).detach()
       
       return quantized, indices, 1 - similarities
   ```

2. **`quantize_to_clusters(subclusters)`**
   ```python
   def quantize_to_clusters(self, subclusters):
       """
       Args:
           subclusters: [B, T, D] quantized vectors from Z
       Returns:
           quantized: [B, T, D] quantized vectors from codebook Q
           indices: [B, T] indices into codebook Q
           distances: [B, T, |Q|] cosine distances
       """
       # L2 normalize
       subclusters = F.normalize(subclusters, dim=-1)
       codebook_q = F.normalize(self.codebook_q, dim=-1)
       
       # Compute cosine similarity
       similarities = torch.matmul(subclusters, codebook_q.t())  # [B, T, |Q|]
       
       # Assign to nearest
       indices = torch.argmax(similarities, dim=-1)  # [B, T]
       
       # Lookup quantized vectors
       quantized = self.codebook_q[indices]  # [B, T, D]
       
       # Straight-through estimator
       quantized = subclusters + (quantized - subclusters).detach()
       
       return quantized, indices, 1 - similarities
   ```

#### Step 2.2: Implement Exponential Moving Average (EMA) Updates
**Functions to Implement**:

1. **`update_codebook_z(embeddings, z_indices)`**
   ```python
   def update_codebook_z(self, embeddings, z_indices):
       """
       EMA update for subcluster codebook Z
       Equation 3 from methodology
       """
       beta = self.beta  # 0.8
       
       # Flatten batch and time dimensions
       embeddings_flat = embeddings.reshape(-1, embeddings.shape[-1])
       indices_flat = z_indices.reshape(-1)
       
       # Count assignments to each prototype
       encodings = F.one_hot(indices_flat, self.num_subclusters).float()  # [B*T, |Z|]
       
       # Update counts: N_hat_zj = beta * N_zj + (1 - beta) * count
       self.cluster_size_z = self.cluster_size_z * beta + \
                              encodings.sum(0) * (1 - beta)
       
       # Sum of embeddings assigned to each prototype
       embed_sum = torch.matmul(encodings.t(), embeddings_flat)  # [|Z|, D]
       
       # Update embedding sum: beta * old_sum + (1 - beta) * new_sum
       self.embed_avg_z = self.embed_avg_z * beta + embed_sum * (1 - beta)
       
       # Update prototypes: z_hat_j = embed_avg_z / cluster_size_z
       n = self.cluster_size_z.unsqueeze(1)
       self.codebook_z.data = self.embed_avg_z / (n + 1e-5)
   ```

2. **`update_codebook_q(subclusters, q_indices)`**
   ```python
   def update_codebook_q(self, subclusters, q_indices):
       """
       EMA update for cluster codebook Q
       Equation 4 from methodology
       """
       beta = self.beta
       
       # Flatten
       subclusters_flat = subclusters.reshape(-1, subclusters.shape[-1])
       indices_flat = q_indices.reshape(-1)
       
       # Count assignments
       encodings = F.one_hot(indices_flat, self.num_clusters).float()
       
       # Update counts
       self.cluster_size_q = self.cluster_size_q * beta + \
                              encodings.sum(0) * (1 - beta)
       
       # Update embedding sum
       embed_sum = torch.matmul(encodings.t(), subclusters_flat)
       self.embed_avg_q = self.embed_avg_q * beta + embed_sum * (1 - beta)
       
       # Update prototypes
       n = self.cluster_size_q.unsqueeze(1)
       self.codebook_q.data = self.embed_avg_q / (n + 1e-5)
   ```

3. **`replace_dead_codes()`**
   ```python
   def replace_dead_codes(self, embeddings, subclusters):
       """
       Replace unused prototypes with random embeddings from batch
       Threshold: N_hat_zj < 3 for Z, N_hat_qi < 1 for Q
       """
       # Check for dead codes in Z
       dead_z = self.cluster_size_z < 3
       if dead_z.any():
           # Sample random embeddings from current batch
           random_indices = torch.randint(0, embeddings.shape[0] * embeddings.shape[1], 
                                         (dead_z.sum(),))
           embeddings_flat = embeddings.reshape(-1, embeddings.shape[-1])
           self.codebook_z.data[dead_z] = embeddings_flat[random_indices]
           self.cluster_size_z[dead_z] = 1.0
       
       # Check for dead codes in Q
       dead_q = self.cluster_size_q < 1
       if dead_q.any():
           subclusters_flat = subclusters.reshape(-1, subclusters.shape[-1])
           random_indices = torch.randint(0, subclusters.shape[0] * subclusters.shape[1],
                                         (dead_q.sum(),))
           self.codebook_q.data[dead_q] = subclusters_flat[random_indices]
           self.cluster_size_q[dead_q] = 1.0
   ```

#### Step 2.3: Complete Forward Pass
**Function**: `forward(actions, training=True)`
```python
def forward(self, actions, training=True):
    """
    Args:
        actions: [B, T, action_dim] input actions
        training: bool, whether in training mode
    Returns:
        dict with keys:
            'embeddings': [B, T, D] encoder output
            'quantized_z': [B, T, D] subcluster quantized
            'quantized_q': [B, T, D] cluster quantized
            'reconstructed_actions': [B, T, action_dim]
            'z_indices': [B, T] subcluster indices
            'q_indices': [B, T] cluster indices
    """
    # 1. Encode
    embeddings = self.encoder(actions)  # [B, T, D]
    
    # 2. Quantize to subclusters (Z)
    quantized_z, z_indices, _ = self.quantize_to_subclusters(embeddings)
    
    # 3. Quantize to clusters (Q)
    quantized_q, q_indices, _ = self.quantize_to_clusters(quantized_z)
    
    # 4. Decode from cluster level
    reconstructed = self.decoder(quantized_q)  # [B, T, action_dim]
    
    # 5. Update codebooks (if training)
    if training:
        with torch.no_grad():
            self.update_codebook_z(embeddings, z_indices)
            self.update_codebook_q(quantized_z, q_indices)
            self.replace_dead_codes(embeddings, quantized_z)
    
    return {
        'embeddings': embeddings,
        'quantized_z': quantized_z,
        'quantized_q': quantized_q,
        'reconstructed_actions': reconstructed,
        'z_indices': z_indices,
        'q_indices': q_indices
    }
```

---

### Phase 3: Loss Functions Implementation

#### Step 3.1: Reconstruction Loss
**Function**: `compute_reconstruction_loss(pred_actions, true_actions)`
```python
def compute_reconstruction_loss(self, pred_actions, true_actions):
    """
    L_rec = sum_t ||x_mt - x_hat_mt||^2_2
    """
    return F.mse_loss(pred_actions, true_actions)
```

#### Step 3.2: Commitment Losses
**Functions**:

1. **`compute_commitment_loss_z(embeddings, quantized_z)`**
   ```python
   def compute_commitment_loss_z(self, embeddings, quantized_z):
       """
       L_commit_z = sum_t ||e_mt - sg[z_mt]||^2_2
       Stop gradient on quantized vectors
       """
       return F.mse_loss(embeddings, quantized_z.detach())
   ```

2. **`compute_commitment_loss_q(quantized_z, quantized_q)`**
   ```python
   def compute_commitment_loss_q(self, quantized_z, quantized_q):
       """
       L_commit_q = sum_t ||z_mt - sg[q_mt]||^2_2
       Stop gradient on cluster prototypes
       """
       return F.mse_loss(quantized_z, quantized_q.detach())
   ```

#### Step 3.3: Combined Loss
**Function**: `compute_vqvae_loss(vqvae_outputs, true_actions, lambda_rec=1.0)`
```python
def compute_vqvae_loss(self, vqvae_outputs, true_actions, lambda_rec=1.0):
    """
    L_total = L_commit_z + L_commit_q + lambda_rec * L_rec
    """
    L_rec = self.compute_reconstruction_loss(
        vqvae_outputs['reconstructed_actions'],
        true_actions
    )
    
    L_commit_z = self.compute_commitment_loss_z(
        vqvae_outputs['embeddings'],
        vqvae_outputs['quantized_z']
    )
    
    L_commit_q = self.compute_commitment_loss_q(
        vqvae_outputs['quantized_z'],
        vqvae_outputs['quantized_q']
    )
    
    total_loss = L_commit_z + L_commit_q + lambda_rec * L_rec
    
    return {
        'vqvae_loss': total_loss,
        'L_rec': L_rec,
        'L_commit_z': L_commit_z,
        'L_commit_q': L_commit_q
    }
```

---

### Phase 4: Integration with ICL Transformer

#### Step 4.1: Modify ICLTransformer Class
**Location**: `robomimic/algo/icl.py`

**Add to `_create_networks()`**:
```python
def _create_networks(self):
    """
    Creates networks and places them into @self.nets.
    """
    assert self.algo_config.transformer.enabled

    self.nets = nn.ModuleDict()
    self.nets["policy"] = PolicyNets.TransformerActorNetwork(
        obs_shapes=self.obs_shapes,
        goal_shapes=self.goal_shapes,
        ac_dim=self.ac_dim,
        encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
            self.obs_config.encoder
        ),
        **BaseNets.transformer_args_from_config(self.algo_config.transformer),
    )
    
    # Add Hierarchical VQ-VAE
    if self.vq_vae_enabled:
        from robomimic.models.hierarchical_vqvae_nets import HierarchicalVQVAE
        
        self.nets["vqvae"] = HierarchicalVQVAE(
            action_dim=self.ac_dim,
            num_subclusters=self.algo_config.transformer.vqvae.num_subclusters,
            num_clusters=self.algo_config.transformer.vqvae.num_clusters,
            embed_dim=self.algo_config.transformer.embed_dim,
            num_stages=self.algo_config.transformer.vqvae.num_stages,
            beta=self.algo_config.transformer.vqvae.beta_ema,
        )
    
    self._set_params_from_config()
    self.nets = self.nets.float().to(self.device)
```

**Add to `_set_params_from_config()`**:
```python
def _set_params_from_config(self):
    """
    Read specific config variables we need for training / eval.
    """
    self.context_length = self.algo_config.transformer.context_length
    self.supervise_all_steps = self.algo_config.transformer.supervise_all_steps
    self.pred_future_acs = self.algo_config.transformer.pred_future_acs
    self.fast_enabled = self.algo_config.transformer.fast_enabled
    self.bin_enabled = self.algo_config.transformer.bin_enabled
    self.vq_vae_enabled = self.algo_config.transformer.vq_vae_enabled
    self.ln_act_enabled = self.algo_config.transformer.ln_act_enabled
    
    if self.vq_vae_enabled:
        self.vqvae_lambda_rec = self.algo_config.transformer.vqvae.lambda_rec
        self.vqvae_pretrain_epochs = self.algo_config.transformer.vqvae.get("pretrain_epochs", 0)
    
    if self.pred_future_acs:
        assert self.supervise_all_steps is True
```

#### Step 4.2: Update Forward Training Pass
**Function**: `_forward_training(self, batch, epoch=None)`

```python
def _forward_training(self, batch, epoch=None):
    """
    Modified forward pass with hierarchical VQ-VAE integration
    """
    # Ensure context length consistency
    TensorUtils.assert_size_at_dim(
        batch["obs"],
        size=(self.context_length),
        dim=1,
        msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(
            self.context_length
        ),
    )

    predictions = OrderedDict()
    
    # Phase 1: VQ-VAE Processing (if enabled)
    if self.vq_vae_enabled:
        # Encode all actions through VQ-VAE
        vqvae_outputs = self.nets["vqvae"](
            actions=batch["actions"],
            training=True
        )
        predictions["vqvae_outputs"] = vqvae_outputs
        
        # Use quantized cluster embeddings as action representation for policy
        # This provides a compressed, tokenized representation
        action_inputs = vqvae_outputs["quantized_q"]  # [B, T, D]
    else:
        action_inputs = None
    
    # Phase 2: Policy Prediction (continuous actions)
    predictions["actions"] = self.nets["policy"](
        obs_dict=batch["obs"], 
        actions=action_inputs,  # Can be None or quantized embeddings
        goal_dict=batch["goal_obs"]
    )
    
    if not self.supervise_all_steps:
        # Only supervise final timestep
        predictions["actions"] = predictions["actions"][:, -1, :]
    
    return predictions
```

#### Step 4.3: Update Loss Computation
**Function**: `_compute_losses(self, predictions, batch)`

```python
def _compute_losses(self, predictions, batch):
    """
    Compute losses for both VQ-VAE and policy
    """
    losses = OrderedDict()
    
    # 1. VQ-VAE Losses (if enabled)
    if self.vq_vae_enabled and "vqvae_outputs" in predictions:
        vqvae_losses = self.nets["vqvae"].compute_vqvae_loss(
            predictions["vqvae_outputs"],
            batch["actions"],
            lambda_rec=self.vqvae_lambda_rec
        )
        losses.update(vqvae_losses)
    
    # 2. Policy Action Prediction Loss (L2)
    a_target = batch["actions"]
    actions = predictions["actions"]
    losses["l2_loss"] = nn.MSELoss()(actions, a_target)
    losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
    losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

    action_losses = [
        self.algo_config.loss.l2_weight * losses["l2_loss"],
        self.algo_config.loss.l1_weight * losses["l1_loss"],
        self.algo_config.loss.cos_weight * losses["cos_loss"],
    ]
    action_loss = sum(action_losses)
    losses["action_loss"] = action_loss
    
    return losses
```

#### Step 4.4: Update Training Step
**Function**: `_train_step(self, losses)`

```python
def _train_step(self, losses):
    """
    Backpropagation with separate handling for VQ-VAE and policy
    """
    info = OrderedDict()
    
    # 1. Train VQ-VAE (if enabled and has loss)
    if self.vq_vae_enabled and 'vqvae_loss' in losses:
        vqvae_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["vqvae"],
            optim=self.optimizers["vqvae"],
            loss=losses["vqvae_loss"],
            max_grad_norm=self.global_config.train.max_grad_norm,
            retain_graph=True  # Keep graph for policy training
        )
        info['vqvae_grad_norms'] = vqvae_grad_norms
    
    # 2. Train Policy Network
    policy_grad_norms = TorchUtils.backprop_for_loss(
        net=self.nets["policy"],
        optim=self.optimizers["policy"],
        loss=losses["action_loss"],
        max_grad_norm=self.global_config.train.max_grad_norm,
    )
    info["policy_grad_norms"] = policy_grad_norms

    # Step schedulers
    for k in self.lr_schedulers:
        if self.lr_schedulers[k] is not None:
            self.lr_schedulers[k].step()
    
    return info
```

---

### Phase 5: Inference & FIFA Integration

#### Step 5.1: Implement Soft Assignment (FIFA)
**Add to `HierarchicalVQVAE` class**:

```python
def compute_soft_assignment(self, embeddings):
    """
    FIFA: Compute soft assignment probabilities using hierarchical similarity
    Equation 8: sim_mt,i = sum_j [(1 - d(e_mt, z_j)) + (1 - d(z_j, q_i))]
    """
    # Normalize
    embeddings = F.normalize(embeddings, dim=-1)
    Z = F.normalize(self.codebook_z, dim=-1)
    Q = F.normalize(self.codebook_q, dim=-1)
    
    # Compute cosine similarities
    sim_e_to_z = torch.matmul(embeddings, Z.t())  # [B, T, |Z|]
    sim_z_to_q = torch.matmul(Z, Q.t())  # [|Z|, |Q|]
    
    # Aggregate similarities for each cluster
    # sim_mt_i = sum_j [sim(e_mt, z_j) + sim(z_j, q_i)]
    sim_mt_i = sim_e_to_z.unsqueeze(-1) + sim_z_to_q.unsqueeze(0).unsqueeze(0)  # [B, T, |Z|, |Q|]
    sim_mt_i = sim_mt_i.sum(dim=2)  # Sum over |Z| -> [B, T, |Q|]
    
    # Softmax for soft assignment
    soft_probs = F.softmax(sim_mt_i, dim=-1)  # [B, T, |Q|]
    
    return soft_probs

def encode(self, actions):
    """Encode actions without training (for inference)"""
    with torch.no_grad():
        embeddings = self.encoder(actions)
        quantized_z, z_indices, _ = self.quantize_to_subclusters(embeddings)
        quantized_q, q_indices, _ = self.quantize_to_clusters(quantized_z)
    
    return {
        'embeddings': embeddings,
        'quantized_z': quantized_z,
        'quantized_q': quantized_q,
        'z_indices': z_indices,
        'q_indices': q_indices
    }
```

#### Step 5.2: Update get_action for Inference
**Function**: `get_action(self, obs_dict, context_batch, goal_dict=None)`

```python
def get_action(self, obs_dict, context_batch, goal_dict=None):
    """
    Get policy action outputs with optional VQ-VAE tokenization
    """
    assert not self.nets.training

    context_obs = context_batch["obs"]
    context_actions = context_batch["actions"]

    # Process context actions through VQ-VAE if enabled
    if self.vq_vae_enabled:
        # Encode context actions
        vqvae_out = self.nets["vqvae"].encode(context_actions)
        
        # Use quantized embeddings as action representation
        action_inputs = vqvae_out['quantized_q']  # [B, T, D]
        
        # Optional: use soft assignment for smoother inference
        # soft_probs = self.nets["vqvae"].compute_soft_assignment(vqvae_out['embeddings'])
    else:
        action_inputs = context_actions

    # Forward through policy
    output = self.nets["policy"](
        obs_dict, 
        context_obs, 
        actions=action_inputs, 
        goal_dict=goal_dict
    )

    # Extract action based on prediction mode
    if self.supervise_all_steps:
        if self.algo_config.transformer.pred_future_acs:
            output = output[:, 0, :]
        else:
            output = output[:, -1, :]
    else:
        output = output[:, -1, :]

    return output
```

---

### Phase 6: Configuration Updates

#### Step 6.1: Update Config JSON
**Add to config**:
```json
{
    "algo_name": "icl",
    "experiment": {
        // ... existing experiment config ...
    },
    "train": {
        // ... existing train config ...
    },
    "algo": {
        "optim_params": {
            "policy": {
                "optimizer_type": "adamw",
                "learning_rate": {
                    "initial": 0.0001,
                    "decay_factor": 1.0,
                    "epoch_schedule": [100],
                    "scheduler_type": "constant_with_warmup"
                },
                "regularization": {
                    "L2": 0.01
                }
            },
            "vqvae": {
                "optimizer_type": "adamw",
                "learning_rate": {
                    "initial": 0.0001,
                    "decay_factor": 1.0,
                    "epoch_schedule": [],
                    "scheduler_type": "constant"
                },
                "regularization": {
                    "L2": 0.0001
                }
            }
        },
        "loss": {
            "l2_weight": 1.0,
            "l1_weight": 0.0,
            "cos_weight": 0.0
        },
        "actor_layer_dims": [],
        "gaussian": {
            "enabled": false
        },
        "gmm": {
            "enabled": false
        },
        "vae": {
            "enabled": false
        },
        "rnn": {
            "enabled": false
        },
        "transformer": {
            "enabled": true,
            "supervise_all_steps": true,
            "pred_future_acs": true,
            "causal": false,
            "num_layers": 6,
            "embed_dim": 512,
            "num_heads": 8,
            "context_length": 10,
            "vq_vae_enabled": true,
            "vqvae": {
                "num_subclusters": 128,
                "num_clusters": 32,
                "embed_dim": 512,
                "num_stages": 2,
                "num_layers_per_stage": 10,
                "lambda_rec": 1.0,
                "beta_ema": 0.8,
                "dead_code_threshold_z": 3,
                "dead_code_threshold_q": 1,
                "pretrain_epochs": 0,
                "use_fifa_inference": false
            }
        }
    },
    "observation": {
        // ... existing observation config ...
    }
}
```

---

### Phase 7: Testing & Validation

#### Step 7.1: Unit Tests

1. **Test MSTCN Architecture**
   ```python
   def test_mstcn_stage