"""
Add this to robomimic/config/icl_config.py or create a new config file
"""

from robomimic.config.base_config import BaseConfig
from robomimic.config.config import Config

# If ICLConfig already exists, you can inherit from it
# Otherwise, define it based on BaseConfig


class ICLHVQVAEConfig(BaseConfig):
    """
    Config for ICL with Hierarchical VQ-VAE
    """

    ALGO_NAME = "icl_hvqvae"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config.
        """
        super(ICLHVQVAEConfig, self).algo_config()

        # Optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4
        self.algo.optim_params.policy.learning_rate.decay_factor = 1.0
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [100]
        self.algo.optim_params.policy.learning_rate.scheduler_type = (
            "constant_with_warmup"
        )
        self.algo.optim_params.policy.regularization.L2 = 0.01

        # VQ-VAE optimizer
        self.algo.optim_params.vqvae = Config()
        self.algo.optim_params.vqvae.optimizer_type = "adamw"
        self.algo.optim_params.vqvae.learning_rate = Config()
        self.algo.optim_params.vqvae.learning_rate.initial = 1e-4
        self.algo.optim_params.vqvae.learning_rate.decay_factor = 1.0
        self.algo.optim_params.vqvae.learning_rate.epoch_schedule = []
        self.algo.optim_params.vqvae.learning_rate.scheduler_type = "constant"
        self.algo.optim_params.vqvae.regularization = Config()
        self.algo.optim_params.vqvae.regularization.L2 = 1e-4

        # Loss weights
        self.algo.loss = Config()
        self.algo.loss.l2_weight = 1.0
        self.algo.loss.l1_weight = 0.0
        self.algo.loss.cos_weight = 0.0

        # Actor
        self.algo.actor_layer_dims = []

        # Gaussian, GMM, VAE, RNN (all disabled)
        self.algo.gaussian = Config()
        self.algo.gaussian.enabled = False

        self.algo.gmm = Config()
        self.algo.gmm.enabled = False

        self.algo.vae = Config()
        self.algo.vae.enabled = False

        self.algo.rnn = Config()
        self.algo.rnn.enabled = False

        # Transformer configuration
        self.algo.transformer = Config()
        self.algo.transformer.enabled = True
        self.algo.transformer.context_length = 10
        self.algo.transformer.supervise_all_steps = True
        self.algo.transformer.pred_future_acs = True
        self.algo.transformer.causal = False
        self.algo.transformer.num_layers = 6
        self.algo.transformer.embed_dim = 512
        self.algo.transformer.num_heads = 8
        self.algo.transformer.fast_enabled = False
        self.algo.transformer.bin_enabled = False
        self.algo.transformer.vq_vae_enabled = True
        self.algo.transformer.ln_act_enabled = True

        # Hierarchical VQ-VAE configuration
        self.algo.transformer.vqvae = Config()
        self.algo.transformer.vqvae.num_subclusters = 256
        self.algo.transformer.vqvae.num_clusters = 128
        self.algo.transformer.vqvae.embed_dim = 512
        self.algo.transformer.vqvae.num_stages = 2
        self.algo.transformer.vqvae.num_layers_per_stage = 10
        self.algo.transformer.vqvae.lambda_rec = 1.0
        self.algo.transformer.vqvae.beta_ema = 0.8
        self.algo.transformer.vqvae.dropout = 0.1
        self.algo.transformer.vqvae.dead_code_threshold_z = 3
        self.algo.transformer.vqvae.dead_code_threshold_q = 1
        self.algo.transformer.vqvae.pretrain_epochs = 0
        self.algo.transformer.vqvae.use_fifa_inference = False

        # Language conditioning
        self.algo.language_conditioned = False


# Register the config
from robomimic.config import config_factory


@config_factory.register_config("icl_hvqvae")
def get_icl_hvqvae_config():
    return ICLHVQVAEConfig()
