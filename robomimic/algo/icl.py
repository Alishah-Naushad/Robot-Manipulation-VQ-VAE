"""
Implementation of In-context learning Behavior Cloning (ICL).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import torch.optim as optim


import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.macros import LANG_EMB_KEY

from robomimic.algo import register_algo_factory_func, PolicyAlgo


# Register the new algo


@register_algo_factory_func("icl_hvqvae")
def algo_config_to_class_hvqvae(algo_config):
    """
    Maps algo config to the ICL Hierarchical VQ-VAE algo class to instantiate.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    algo_class, algo_kwargs = ICLTransformerHVQVAE, {}
    return algo_class, algo_config


@register_algo_factory_func("icl")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the ICL algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    # note: we need the check below because some configs import BCConfig and exclude
    # some of these options
    gaussian_enabled = "gaussian" in algo_config and algo_config.gaussian.enabled
    gmm_enabled = "gmm" in algo_config and algo_config.gmm.enabled
    vae_enabled = "vae" in algo_config and algo_config.vae.enabled

    rnn_enabled = algo_config.rnn.enabled
    transformer_enabled = algo_config.transformer.enabled
    print(transformer_enabled)
    print("*" * 25)
    # input('cree')
    if gaussian_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = ICLGaussian, {}
    elif gmm_enabled:
        if rnn_enabled:
            algo_class, algo_kwargs = ICLRNN_GMM, {}
        elif transformer_enabled:
            print("here2")
            print("*" * 25)
            algo_class, algo_kwargs = ICLTransformerHVQVAE, {}
        else:
            algo_class, algo_kwargs = ICLGMM, {}
    elif vae_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            print("here vae")
            print("*" * 25)
            algo_class, algo_kwargs = ICLTransformerHVQVAE, {}
    else:
        if rnn_enabled:
            algo_class, algo_kwargs = ICLRNN, {}
        elif transformer_enabled:
            print("here")
            print("*" * 25)
            algo_class, algo_kwargs = ICLTransformerHVQVAE, {}
        else:
            algo_class, algo_kwargs = ICL, {}

    return algo_class, algo_kwargs


class ICL(PolicyAlgo):
    """
    Normal ICL training.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.ActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
                self.obs_config.encoder
            ),
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get(
            "goal_obs", None
        )  # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]
        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(ICL, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _forward_training(self, batch):
        """
        Internal helper function for ICL algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()
        actions = self.nets["policy"](
            obs_dict=batch["obs"], goal_dict=batch["goal_obs"]
        )
        predictions["actions"] = actions
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for ICL algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses

    def _train_step(self, losses):
        """
        Internal helper function for ICL algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
            max_grad_norm=self.global_config.train.max_grad_norm,
        )
        info["policy_grad_norms"] = policy_grad_norms

        # step through optimizers
        for k in self.lr_schedulers:
            if self.lr_schedulers[k] is not None:
                self.lr_schedulers[k].step()
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(ICL, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)


"""
Modified ICL Transformer class with Hierarchical VQ-VAE integration
Add this to robomimic/algo/icl.py after the existing ICLTransformer class
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils


class ICLGaussian(ICL):
    """
    ICL training with a Gaussian policy.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gaussian.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GaussianActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            fixed_std=self.algo_config.gaussian.fixed_std,
            init_std=self.algo_config.gaussian.init_std,
            std_limits=(self.algo_config.gaussian.min_std, 7.5),
            std_activation=self.algo_config.gaussian.std_activation,
            low_noise_eval=self.algo_config.gaussian.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
                self.obs_config.encoder
            ),
        )

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for ICL algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 1
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for ICL algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class ICLGMM(ICLGaussian):
    """
    ICL training with a Gaussian Mixture Model policy.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
                self.obs_config.encoder
            ),
        )

        self.nets = self.nets.float().to(self.device)


class ICLVAE(ICL):
    """
    ICL training with a VAE policy.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.VAEActor(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            device=self.device,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
                self.obs_config.encoder
            ),
            **VAENets.vae_args_from_config(self.algo_config.vae),
        )

        self.nets = self.nets.float().to(self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Update from superclass to set categorical temperature, for categorical VAEs.
        """
        if self.algo_config.vae.prior.use_categorical:
            temperature = (
                self.algo_config.vae.prior.categorical_init_temp
                - epoch * self.algo_config.vae.prior.categorical_temp_anneal_step
            )
            temperature = max(
                temperature, self.algo_config.vae.prior.categorical_min_temp
            )
            self.nets["policy"].set_gumbel_temperature(temperature)
        return super(ICLVAE, self).train_on_batch(batch, epoch, validate=validate)

    def _forward_training(self, batch):
        """
        Internal helper function for ICL algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        vae_inputs = dict(
            actions=batch["actions"],
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
            freeze_encoder=batch.get("freeze_encoder", False),
        )

        vae_outputs = self.nets["policy"].forward_train(**vae_inputs)
        predictions = OrderedDict(
            actions=vae_outputs["decoder_outputs"],
            kl_loss=vae_outputs["kl_loss"],
            reconstruction_loss=vae_outputs["reconstruction_loss"],
            encoder_z=vae_outputs["encoder_z"],
        )
        if not self.algo_config.vae.prior.use_categorical:
            with torch.no_grad():
                encoder_variance = torch.exp(vae_outputs["encoder_params"]["logvar"])
            predictions["encoder_variance"] = encoder_variance
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for ICL algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # total loss is sum of reconstruction and KL, weighted by beta
        kl_loss = predictions["kl_loss"]
        recons_loss = predictions["reconstruction_loss"]
        action_loss = recons_loss + self.algo_config.vae.kl_weight * kl_loss
        return OrderedDict(
            recons_loss=recons_loss,
            kl_loss=kl_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["KL_Loss"] = info["losses"]["kl_loss"].item()
        log["Reconstruction_Loss"] = info["losses"]["recons_loss"].item()
        if self.algo_config.vae.prior.use_categorical:
            log["Gumbel_Temperature"] = self.nets["policy"].get_gumbel_temperature()
        else:
            log["Encoder_Variance"] = (
                info["predictions"]["encoder_variance"].mean().item()
            )
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class ICLRNN(ICL):
    """
    ICL training with an RNN policy.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
                self.obs_config.encoder
            ),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]
        input_batch["goal_obs"] = batch.get(
            "goal_obs", None
        )  # goals may not be present
        input_batch["actions"] = batch["actions"]

        if self._rnn_is_open_loop:
            # replace the observation sequence with one that only consists of the first observation.
            # This way, all actions are predicted "open-loop" after the first observation, based
            # on the rnn hidden state.
            n_steps = batch["actions"].shape[1]
            obs_seq_start = TensorUtils.index_at_time(batch["obs"], ind=0)
            input_batch["obs"] = TensorUtils.unsqueeze_expand_at(
                obs_seq_start, size=n_steps, dim=1
            )

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(
                batch_size=batch_size, device=self.device
            )

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state
        )
        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._rnn_hidden_state = None
        self._rnn_counter = 0


class ICLRNN_GMM(ICLRNN):
    """
    ICL training with an RNN GMM policy.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.rnn.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
                self.obs_config.encoder
            ),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for ICL algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2  # [B, T]
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for ICL algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class ICLTransformer(ICL):
    """
    ICL training with a Transformer policy.
    """

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
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)

    def _set_params_from_config(self):
        """
        Read specific config variables we need for training / eval.
        Called by @_create_networks method
        """
        self.context_length = self.algo_config.transformer.context_length
        self.supervise_all_steps = self.algo_config.transformer.supervise_all_steps
        self.pred_future_acs = self.algo_config.transformer.pred_future_acs
        self.fast_enabled = self.algo_config.transformer.fast_enabled
        self.bin_enabled = self.algo_config.transformer.bin_enabled
        self.vq_vae_enabled = self.algo_config.transformer.vq_vae_enabled
        self.ln_act_enabled = self.algo_config.transformer.ln_act_enabled
        # self.action_input_shape = self.algo_config.transformer.action_input_shape
        if self.pred_future_acs:
            assert self.supervise_all_steps is True

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        h = self.context_length
        input_batch["obs"] = {k: batch["obs"][k][:, :h, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get(
            "goal_obs", None
        )  # goals may not be present

        if self.supervise_all_steps:
            # supervision on entire sequence (instead of just current timestep)
            if self.pred_future_acs:
                ac_start = h - 1
            else:
                ac_start = 0
            input_batch["actions"] = batch["actions"][:, ac_start : ac_start + h, :]
        else:
            # just use current timestep
            input_batch["actions"] = batch["actions"][:, h - 1, :]

        if self.pred_future_acs:
            assert input_batch["actions"].shape[1] == h

        input_batch = TensorUtils.to_device(
            TensorUtils.to_float(input_batch), self.device
        )
        return input_batch

    def _forward_training(self, batch, epoch=None):
        """
        Internal helper function for ICLTransformer algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        print("in icl transformer training")
        print("*" * 25)
        TensorUtils.assert_size_at_dim(
            batch["obs"],
            size=(self.context_length),
            dim=1,
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(
                self.context_length
            ),
        )

        predictions = OrderedDict()
        predictions["actions"] = self.nets["policy"](
            obs_dict=batch["obs"], actions=None, goal_dict=batch["goal_obs"]
        )
        if not self.supervise_all_steps:
            # only supervise final timestep
            predictions["actions"] = predictions["actions"][:, -1, :]
        return predictions

    def get_action(self, obs_dict, context_batch, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        context_obs = context_batch["obs"]
        context_action = context_batch["actions"]

        output = self.nets["policy"](
            obs_dict, context_obs, actions=context_action, goal_dict=goal_dict
        )

        if self.supervise_all_steps:
            if self.algo_config.transformer.pred_future_acs:
                output = output[:, 0, :]
            else:
                output = output[:, -1, :]
        else:
            output = output[:, -1, :]

        return output


class ICLTransformerHVQVAE(ICLTransformer):
    """
    ICL Transformer with Hierarchical VQ-VAE for action tokenization
    Extends ICLTransformer to add two-level action quantization
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        Adds hierarchical VQ-VAE if enabled.
        """
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()

        # Main transformer policy network
        transformer_args = BaseNets.transformer_args_from_config(
            self.algo_config.transformer
        )
        if "transformer_fast_enabled" in transformer_args:
            transformer_args.pop("transformer_fast_enabled")
        if "transformer_fast_config" in transformer_args:
            transformer_args.pop("transformer_fast_config")
        if "transformer_bin_enabled" in transformer_args:
            transformer_args.pop("transformer_bin_enabled")
        if "transformer_vq_vae_enabled" in transformer_args:
            transformer_args.pop("transformer_vq_vae_enabled")
        if "transformer_ln_act_enabled" in transformer_args:
            transformer_args.pop("transformer_ln_act_enabled")
        self.nets["policy"] = PolicyNets.TransformerActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
                self.obs_config.encoder
            ),
            **transformer_args,
        )

        # Add Hierarchical VQ-VAE if enabled
        if self.algo_config.transformer.get("vq_vae_enabled", False):
            from robomimic.algo.heirarchical_vqvae import HierarchicalVQVAE

            vqvae_config = self.algo_config.transformer.vqvae

            self.nets["vqvae"] = HierarchicalVQVAE(
                action_dim=self.ac_dim,
                num_subclusters=vqvae_config.get("num_subclusters", 128),
                num_clusters=vqvae_config.get("num_clusters", 32),
                embed_dim=vqvae_config.get(
                    "embed_dim", self.algo_config.transformer.embed_dim
                ),
                num_stages=vqvae_config.get("num_stages", 2),
                num_layers_per_stage=vqvae_config.get("num_layers_per_stage", 10),
                beta=vqvae_config.get("beta_ema", 0.8),
                dropout=vqvae_config.get("dropout", 0.1),
                kmeans_init=True,
            )

            print(f"[ICLTransformerHVQVAE] Created Hierarchical VQ-VAE:")
            print(f"  - Subclusters (Z): {vqvae_config.get('num_subclusters', 128)}")
            print(f"  - Clusters (Q): {vqvae_config.get('num_clusters', 32)}")
            print(
                f"  - Embed dim: {vqvae_config.get('embed_dim', self.algo_config.transformer.embed_dim)}"
            )

        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)

    def _set_params_from_config(self):
        """
        Read specific config variables we need for training / eval.
        Called by @_create_networks method
        """
        self.context_length = self.algo_config.transformer.context_length
        self.supervise_all_steps = self.algo_config.transformer.supervise_all_steps
        self.pred_future_acs = self.algo_config.transformer.pred_future_acs
        self.fast_enabled = self.algo_config.transformer.fast_enabled
        self.bin_enabled = self.algo_config.transformer.bin_enabled
        self.vq_vae_enabled = self.algo_config.transformer.get("vq_vae_enabled", False)
        self.ln_act_enabled = self.algo_config.transformer.ln_act_enabled

        # VQ-VAE specific parameters
        if self.vq_vae_enabled:
            vqvae_config = self.algo_config.transformer.vqvae
            self.vqvae_lambda_rec = vqvae_config.get("lambda_rec", 1.0)
            self.vqvae_pretrain_epochs = vqvae_config.get("pretrain_epochs", 0)
            self.vqvae_use_fifa = vqvae_config.get("use_fifa_inference", False)
            print(f"[ICLTransformerHVQVAE] VQ-VAE config:")
            print(f"  - Lambda rec: {self.vqvae_lambda_rec}")
            print(f"  - Pretrain epochs: {self.vqvae_pretrain_epochs}")
            print(f"  - Use FIFA inference: {self.vqvae_use_fifa}")

        if self.pred_future_acs:
            assert self.supervise_all_steps is True

    def _create_optimizers(self):
        """
        Create optimizers for policy and VQ-VAE (if enabled)
        """
        self.optimizers = dict()
        self.lr_schedulers = dict()

        # Policy optimizer
        self.optimizers["policy"] = self._get_optimizer(
            net=self.nets["policy"], optim_params=self.algo_config.optim_params.policy
        )

        # VQ-VAE optimizer (if enabled)
        if self.vq_vae_enabled and "vqvae" in self.nets:
            # Check if separate VQ-VAE optimizer config exists
            if hasattr(self.algo_config.optim_params, "vqvae"):
                vqvae_optim_params = self.algo_config.optim_params.vqvae
            else:
                # Use policy optimizer config as fallback
                vqvae_optim_params = self.algo_config.optim_params.policy

            self.optimizers["vqvae"] = self._get_optimizer(
                net=self.nets["vqvae"], optim_params=vqvae_optim_params
            )

            print(f"[ICLTransformerHVQVAE] Created separate optimizer for VQ-VAE")

    def _get_optimizer(self, net, optim_params):
        """
        Helper to create optimizer from config
        """
        optimizer_type = optim_params.get("optimizer_type", "adam")
        lr = optim_params.learning_rate.initial
        weight_decay = optim_params.regularization.get("L2", 0.0)

        if optimizer_type == "adam":
            return optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "adamw":
            return optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def _forward_training(self, batch, epoch=None):
        """
        Internal helper function for ICLTransformer algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Modified to integrate hierarchical VQ-VAE.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
            epoch (int): current training epoch

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        # Ensure that transformer context length is consistent with temporal dimension of observations
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
        action_inputs = None
        if self.vq_vae_enabled:
            # Forward through VQ-VAE
            vqvae_outputs = self.nets["vqvae"](
                actions=batch["actions"],
                training=self.nets["vqvae"].training,  # Use module's training state
            )

            # Store VQ-VAE outputs for loss computation
            predictions["vqvae_outputs"] = vqvae_outputs

            # Use quantized cluster embeddings as action representation
            # This provides a compressed, tokenized representation of actions
            action_inputs = vqvae_outputs["quantized_q"]  # [B, T, D]

            # Optional: can also use cluster indices directly for discrete tokens
            # action_indices = vqvae_outputs["q_indices"]  # [B, T]

        # Phase 2: Transformer Policy Prediction (continuous actions)
        predictions["actions"] = self.nets["policy"](
            obs_dict=batch["obs"],
            actions=action_inputs,  # None if VQ-VAE disabled, or quantized embeddings
            goal_dict=batch["goal_obs"],
        )

        if not self.supervise_all_steps:
            # Only supervise final timestep
            predictions["actions"] = predictions["actions"][:, -1, :]

        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for ICLTransformer algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()

        # 1. VQ-VAE Losses (if enabled and outputs present)
        if self.vq_vae_enabled and "vqvae_outputs" in predictions:
            vqvae_losses = self.nets["vqvae"].compute_vqvae_loss(
                predictions["vqvae_outputs"],
                batch["actions"],
                lambda_rec=self.vqvae_lambda_rec,
            )

            # Add all VQ-VAE losses
            losses.update(vqvae_losses)

        # 2. Policy Action Prediction Losses
        a_target = batch["actions"]
        actions = predictions["actions"]

        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)

        # Cosine direction loss on eef delta position (first 3 dims)
        if actions.shape[-1] >= 3 and a_target.shape[-1] >= 3:
            losses["cos_loss"] = LossUtils.cosine_loss(
                actions[..., :3], a_target[..., :3]
            )
        else:
            losses["cos_loss"] = torch.tensor(0.0, device=actions.device)

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss

        return losses

    def _train_step(self, losses):
        """
        Co-training: VQ-VAE and Policy optimize a shared combined loss.

        Args:
            losses (dict): dictionary of losses from _compute_losses
        Returns:
            info (OrderedDict): gradient norms and other metrics
        """
        info = OrderedDict()

        # -----------------------------------
        # Case 1: VQ-VAE enabled -> combined loss
        # -----------------------------------
        if self.vq_vae_enabled and "vqvae_loss" in losses:
            combined_loss = 0.1 * losses["vqvae_loss"] + 1.0 * losses["action_loss"]

            # ----- Backward once -----
            combined_loss.backward()

            # ----- VQ-VAE optimizer step -----
            vqvae_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.nets["vqvae"].parameters(), self.global_config.train.max_grad_norm
            )
            self.optimizers["vqvae"].step()
            self.optimizers["vqvae"].zero_grad()

            # ----- Policy optimizer step -----
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.nets["policy"].parameters(), self.global_config.train.max_grad_norm
            )
            self.optimizers["policy"].step()
            self.optimizers["policy"].zero_grad()

            info["vqvae_grad_norm"] = float(vqvae_grad_norm)
            info["policy_grad_norm"] = float(policy_grad_norm)

        # -----------------------------------
        # Case 2: VQ-VAE disabled -> train policy only
        # -----------------------------------
        else:
            policy_grad_norm = TorchUtils.backprop_for_loss(
                net=self.nets["policy"],
                optim=self.optimizers["policy"],
                loss=losses["action_loss"],
                max_grad_norm=self.global_config.train.max_grad_norm,
            )
            info["policy_grad_norm"] = float(policy_grad_norm)

        # -----------------------------------
        # Step schedulers
        # -----------------------------------
        for k in self.lr_schedulers:
            if self.lr_schedulers[k] is not None:
                self.lr_schedulers[k].step()

        return info

    def _train_step_old(self, losses):
        """
        Internal helper function for ICLTransformer algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """
        info = OrderedDict()

        # 1. Train VQ-VAE (if enabled and has loss)
        if (
            self.vq_vae_enabled
            and "vqvae_loss" in losses
            and "vqvae" in self.optimizers
        ):
            vqvae_grad_norms = TorchUtils.backprop_for_loss(
                net=self.nets["vqvae"],
                optim=self.optimizers["vqvae"],
                loss=losses["vqvae_loss"],
                max_grad_norm=self.global_config.train.max_grad_norm,
                retain_graph=True,  # Keep graph for policy training
            )
            info["vqvae_grad_norms"] = vqvae_grad_norms

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

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(ICLTransformer, self).log_info(info)

        # Policy losses
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]

        # VQ-VAE losses (if enabled)
        if self.vq_vae_enabled and "vqvae_loss" in info["losses"]:
            log["VQ-VAE/Total_Loss"] = info["losses"]["vqvae_loss"].item()
            log["VQ-VAE/Reconstruction_Loss"] = info["losses"]["L_rec"].item()
            log["VQ-VAE/Commitment_Z_Loss"] = info["losses"]["L_commit_z"].item()
            log["VQ-VAE/Commitment_Q_Loss"] = info["losses"]["L_commit_q"].item()

            if "vqvae_grad_norms" in info:
                log["VQ-VAE/Grad_Norms"] = info["vqvae_grad_norms"]

            # Codebook utilization statistics
            if hasattr(self.nets["vqvae"], "get_codebook_usage"):
                usage = self.nets["vqvae"].get_codebook_usage()
                log["VQ-VAE/Z_Utilization_pct"] = usage["z_usage_pct"]
                log["VQ-VAE/Q_Utilization_pct"] = usage["q_usage_pct"]
                log["VQ-VAE/Z_Utilization"] = usage["z_used"]
                log["VQ-VAE/Q_Utilization"] = usage["q_used"]
                log["VQ-VAE/Z_Dead_Codes"] = usage["z_dead"]
                log["VQ-VAE/Q_Dead_Codes"] = usage["q_dead"]

        return log

    def get_action(self, obs_dict, context_batch, goal_dict=None):
        """
        Get policy action outputs with optional VQ-VAE tokenization.

        Args:
            obs_dict (dict): current observation
            context_batch (dict): context observations and actions
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        context_obs = context_batch["obs"]
        context_actions = context_batch["actions"]

        # Process context actions through VQ-VAE if enabled
        action_inputs = None
        if self.vq_vae_enabled:
            # Encode context actions
            vqvae_out = self.nets["vqvae"].encode(context_actions)

            # Use quantized cluster embeddings as action representation
            action_inputs = vqvae_out["quantized_q"]  # [B, T, D]

            # Optional: use FIFA soft assignment for smoother inference
            if self.vqvae_use_fifa:
                soft_probs = self.nets["vqvae"].compute_soft_assignment(
                    vqvae_out["embeddings"]
                )
                # Can use soft_probs for weighted combination of cluster embeddings
                # For now, we use hard assignment (quantized_q)

        # Forward through transformer policy
        output = self.nets["policy"](
            obs_dict, actions=context_actions, goal_dict=goal_dict
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

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch. Can be used for epoch-specific logic.

        Args:
            epoch (int): current epoch number
        """
        super(ICLTransformer, self).on_epoch_end(epoch)

        # Log VQ-VAE codebook statistics
        if self.vq_vae_enabled and hasattr(self.nets["vqvae"], "get_codebook_usage"):
            usage = self.nets["vqvae"].get_codebook_usage()
            print(f"\n[Epoch {epoch}] VQ-VAE Codebook Usage:")
            print(f"  - Subcluster (Z) utilization: {usage['z_utilization']*100:.1f}%")
            print(f"  - Cluster (Q) utilization: {usage['q_utilization']*100:.1f}%")
            print(f"  - Dead codes Z: {usage['z_dead']}")
            print(f"  - Dead codes Q: {usage['q_dead']}")

    def serialize(self):
        """
        Serialize model to dictionary for saving.
        Includes VQ-VAE state if enabled.
        """
        state_dict = super(ICLTransformer, self).serialize()

        # Add VQ-VAE specific info
        if self.vq_vae_enabled:
            state_dict["vq_vae_enabled"] = True
            state_dict["vqvae_config"] = {
                "num_subclusters": self.nets["vqvae"].num_subclusters,
                "num_clusters": self.nets["vqvae"].num_clusters,
                "embed_dim": self.nets["vqvae"].embed_dim,
                "beta": self.nets["vqvae"].beta,
            }

        return state_dict

    def deserialize(self, state_dict):
        """
        Load model from serialized state.
        Handles VQ-VAE state if present.
        """
        # Check if VQ-VAE was enabled in saved model
        if state_dict.get("vq_vae_enabled", False):
            if not self.vq_vae_enabled:
                print(
                    "[Warning] Saved model has VQ-VAE but current config doesn't. Enable vq_vae in config."
                )

        return super(ICLTransformer, self).deserialize(state_dict)


class ICLTransformer_GMM(ICLTransformer):
    """
    ICL training with a Transformer GMM policy.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.ICLTransformerGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
                self.obs_config.encoder
            ),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)

        if self.vq_vae_enabled:
            self.vq_vae_model = self.nets["policy"].vq_vae_model
            self.vq_optimizer = optim.AdamW(
                self.vq_vae_model.parameters(), lr=1e-3, weight_decay=1e-4
            )  # Adjust lr and weight_decay as needed

    def _forward_training(self, batch, epoch=None):
        """
        Modify from super class to support GMM training.
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"],
            size=(self.context_length),
            dim=1,
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(
                self.context_length
            ),
        )
        # Split the observation into halves
        mid = batch["obs"]["lang_emb"].shape[0] // 2
        # Split observations
        context_obs = {key: value[:mid] for key, value in batch["obs"].items()}
        train_obs = {key: value[mid:] for key, value in batch["obs"].items()}

        # Split actions
        context_actions, train_actions = batch["actions"][:mid], batch["actions"][mid:]

        if self.vq_vae_enabled:
            self.vq_optimizer.zero_grad()

        dists = self.nets["policy"].forward_train(
            obs_dict=train_obs,
            context_obs=context_obs,
            actions=context_actions,
            goal_dict=batch["goal_obs"],
            low_noise_eval=False,
        )

        if self.vq_vae_enabled:
            self._vq_vae_loss = self.nets["policy"]._vq_vae_loss

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2  # [B, T]

        if not self.supervise_all_steps:
            # only use final timestep prediction by making a new distribution with only final timestep.
            # This essentially does `dists = dists[:, -1]`
            component_distribution = D.Normal(
                loc=dists.component_distribution.base_dist.loc[:, -1],
                scale=dists.component_distribution.base_dist.scale[:, -1],
            )
            component_distribution = D.Independent(component_distribution, 1)
            mixture_distribution = D.Categorical(
                logits=dists.mixture_distribution.logits[:, -1]
            )
            dists = D.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution,
            )

        log_probs = dists.log_prob(train_actions)
        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for ICLTransformer_GMM algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()

        if self.vq_vae_enabled:
            self._vq_vae_loss.backward()
            self.vq_optimizer.step()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
