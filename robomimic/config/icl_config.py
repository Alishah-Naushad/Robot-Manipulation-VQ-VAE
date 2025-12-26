"""
Fixed ICL Config with Hierarchical VQ-VAE support
Replace the relevant sections in robomimic/config/icl_config.py
"""

from robomimic.config.base_config import BaseConfig
from robomimic.config.config import Config


class ICLConfig(BaseConfig):
    ALGO_NAME = "icl"

    def train_config(self):
        """
        ICL algorithms don't need "next_obs" from hdf5 - so save on storage and compute by disabling it.
        """
        super(ICLConfig, self).train_config()
        self.train.hdf5_load_next_obs = False

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """

        # optimization parameters for policy
        self.algo.optim_params.policy.optimizer_type = "adam"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1
        self.algo.optim_params.policy.learning_rate.epoch_schedule = []
        self.algo.optim_params.policy.learning_rate.scheduler_type = (
            "constant_with_warmup"
        )
        self.algo.optim_params.policy.regularization.L2 = 0.00

        # optimization parameters for vqvae (MUST be added before locking)
        self.algo.optim_params.vqvae = Config()
        self.algo.optim_params.vqvae.optimizer_type = "adamw"
        self.algo.optim_params.vqvae.learning_rate = Config()
        self.algo.optim_params.vqvae.learning_rate.initial = 1e-4
        self.algo.optim_params.vqvae.learning_rate.decay_factor = 1.0
        self.algo.optim_params.vqvae.learning_rate.epoch_schedule = []
        self.algo.optim_params.vqvae.learning_rate.scheduler_type = "constant"
        self.algo.optim_params.vqvae.regularization = Config()
        self.algo.optim_params.vqvae.regularization.L2 = 1e-4

        # loss weights
        self.algo.loss.l2_weight = 1.0
        self.algo.loss.l1_weight = 0.0
        self.algo.loss.cos_weight = 0.0

        # MLP network architecture
        self.algo.actor_layer_dims = (1024, 1024)

        # stochastic Gaussian policy settings
        self.algo.gaussian.enabled = False
        self.algo.gaussian.fixed_std = False
        self.algo.gaussian.init_std = 0.1
        self.algo.gaussian.min_std = 0.01
        self.algo.gaussian.std_activation = "softplus"
        self.algo.gaussian.low_noise_eval = True

        # stochastic GMM policy settings
        self.algo.gmm.enabled = True
        self.algo.gmm.num_modes = 5
        self.algo.gmm.min_std = 0.0001
        self.algo.gmm.std_activation = "softplus"
        self.algo.gmm.low_noise_eval = True

        # stochastic VAE policy settings
        self.algo.vae.enabled = False
        self.algo.vae.latent_dim = 14
        self.algo.vae.latent_clip = None
        self.algo.vae.kl_weight = 1.0

        # VAE decoder settings
        self.algo.vae.decoder.is_conditioned = True
        self.algo.vae.decoder.reconstruction_sum_across_elements = False

        # VAE prior settings
        self.algo.vae.prior.learn = False
        self.algo.vae.prior.is_conditioned = False
        self.algo.vae.prior.use_gmm = False
        self.algo.vae.prior.gmm_num_modes = 10
        self.algo.vae.prior.gmm_learn_weights = False
        self.algo.vae.prior.use_categorical = False
        self.algo.vae.prior.categorical_dim = 10
        self.algo.vae.prior.categorical_gumbel_softmax_hard = False
        self.algo.vae.prior.categorical_init_temp = 1.0
        self.algo.vae.prior.categorical_temp_anneal_step = 0.001
        self.algo.vae.prior.categorical_min_temp = 0.3

        self.algo.vae.encoder_layer_dims = (300, 400)
        self.algo.vae.decoder_layer_dims = (300, 400)
        self.algo.vae.prior_layer_dims = (300, 400)

        # RNN policy settings
        self.algo.rnn.enabled = False
        self.algo.rnn.horizon = 10
        self.algo.rnn.hidden_dim = 400
        self.algo.rnn.rnn_type = "LSTM"
        self.algo.rnn.num_layers = 2
        self.algo.rnn.open_loop = False
        self.algo.rnn.kwargs.bidirectional = False
        self.algo.rnn.kwargs.do_not_lock_keys()

        # Transformer policy settings
        self.algo.transformer.enabled = True
        self.algo.transformer.context_length = 16
        self.algo.transformer.embed_dim = 512
        self.algo.transformer.num_layers = 6
        self.algo.transformer.num_heads = 8
        self.algo.transformer.emb_dropout = 0.1
        self.algo.transformer.attn_dropout = 0.1
        self.algo.transformer.block_output_dropout = 0.1
        self.algo.transformer.sinusoidal_embedding = False
        self.algo.transformer.activation = "gelu"
        self.algo.transformer.fast_enabled = False
        self.algo.transformer.bin_enabled = False
        self.algo.transformer.vq_vae_enabled = True
        self.algo.transformer.ln_act_enabled = True
        self.algo.transformer.supervise_all_steps = True
        self.algo.transformer.nn_parameter_for_timesteps = True
        self.algo.transformer.pred_future_acs = True
        self.algo.transformer.causal = False

        # Hierarchical VQ-VAE settings
        self.algo.transformer.vqvae = Config()
        self.algo.transformer.vqvae.num_subclusters = 128
        self.algo.transformer.vqvae.num_clusters = 64
        self.algo.transformer.vqvae.embed_dim = 512
        self.algo.transformer.vqvae.num_stages = 2
        self.algo.transformer.vqvae.num_layers_per_stage = 4
        self.algo.transformer.vqvae.lambda_rec = 0.002
        self.algo.transformer.vqvae.beta_ema = 0.8
        self.algo.transformer.vqvae.dropout = 0.1
        self.algo.transformer.vqvae.dead_code_threshold_z = 2
        self.algo.transformer.vqvae.dead_code_threshold_q = 1
        self.algo.transformer.vqvae.pretrain_epochs = 0
        self.algo.transformer.vqvae.use_fifa_inference = True
        self.algo.transformer.vqvae.do_not_lock_keys()

        self.algo.language_conditioned = False
