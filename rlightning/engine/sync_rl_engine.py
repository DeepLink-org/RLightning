"""Synchronous reinforcement learning engine module.

This module implements the SyncRLEngine for synchronous training where
rollout and training happen sequentially in a single thread.
"""

from typing import Optional

import tree

from rlightning.buffer import DataBuffer
from rlightning.env import EnvGroup
from rlightning.policy import PolicyGroup
from rlightning.utils.config import MainConfig
from rlightning.utils.logger import get_logger, log_metric
from rlightning.utils.registry import ENGINE
from rlightning.utils.utils import InternalFlag

from .base_engine import BaseEngine

logger = get_logger(__name__)


@ENGINE.register("syncrl")
class SyncRLEngine(BaseEngine):
    """Synchronous reinforcement learning engine.

    This engine implements synchronous training where rollout and training
    happen sequentially. Supports both on-policy and off-policy algorithms.

    """

    def __init__(
        self,
        config: MainConfig,
        env_group: EnvGroup = None,
        policy_group: Optional[PolicyGroup] = None,
        buffer: Optional[DataBuffer] = None,
    ) -> None:
        """Initialize the synchronous RL engine."""
        super().__init__(
            config=config,
            env_group=env_group,
            policy_group=policy_group,
            buffer=buffer,
        )

        # init env
        env_meta_list = self.env_group.init()
        self.env_meta_list = env_meta_list

        # init train policy
        self.policy_group.init_train(self.config.train, env_meta_list[0])
        if self.config.cluster.enable_offload:
            self.policy_group.offload_model_param_and_grad(offload_grad=True, offload_optimizer=True)

        # init eval policy after to avoid cuda out of memory
        self.policy_group.init_eval(env_meta=env_meta_list[0])

        # init replay buffer
        env_ids = self.env_group.env_ids
        self.buffer.init(env_meta_list, env_ids)

        if InternalFlag.DEBUG:
            self.warm_up()
        else:
            self.sync_weights()

        self.policy_group.verify_eval_weight_consistency()

    def warm_up(self):
        """init and dummy run for constructing RL dataflow"""
        env_meta_list = self.env_meta_list

        # dummy run rollout (synchronous)
        batched_policy_resp = None
        warm_up_rollout_steps = self.config.train.get("warm_up_rollout_steps", 10)
        for _ in range(warm_up_rollout_steps + 1):
            if batched_policy_resp is None:
                batched_env_ret, _ = self.env_group.reset(
                    seed=0,
                    options={"obj_set": "train"},
                )
            else:
                batched_env_ret, _ = self.env_group.step(batched_policy_resp)

            batched_policy_resp = self.policy_group.rollout_batch(batched_env_ret)
            self.buffer.add_batched_transition(batched_env_ret, batched_policy_resp)

        if self.config.cluster.enable_offload:
            self.policy_group.offload_eval_model()
            self.env_group.offload()

        self.buffer.truncate_episodes(batched_policy_resp.ids())

        # dummy run train
        data = self.buffer.sample(batch_size=self.config.train.batch_size)
        self.policy_group.update_dataset(data)

        self._pre_train_hook()
        self.policy_group.train()
        self._post_train_hook()

        if InternalFlag.DEBUG:
            self.print_timing_summary(reset=True)

        # clear buffer
        self.buffer.clear()
        # reset training state after dummy run
        seed = getattr(self.config.train, "seed", None)
        self.policy_group.reset_training_state(self.config.train, env_meta_list[0], seed=seed)

        self._pre_sync_weights_hook()
        self.sync_weights()
        self._post_sync_weights_hook()

        logger.info("Warm up done, ready to run.")

    def rollout(self, obj_set: str, prefix: str = "") -> None:
        """Perform rollout to collect experience from environments.

        Args:
            obj_set: Object set identifier for environment reset options.
            prefix: Prefix for logging metrics.
        """
        max_rollout_steps = self.config.train.max_rollout_steps
        rollout_epoch = self.config.train.get("rollout_epoch", 1)
        for _ in range(rollout_epoch):
            batched_policy_resp = None
            with self.env_group.auto_reset(max_episode_steps=max_rollout_steps):
                for _ in range(max_rollout_steps + 1):
                    if batched_policy_resp is None:
                        batched_env_ret, truncations = self.env_group.reset(
                            options={"obj_set": obj_set},
                        )
                    else:
                        batched_env_ret, truncations = self.env_group.step(batched_policy_resp)
                    batched_policy_resp = self.policy_group.rollout_batch(batched_env_ret)
                    self.buffer.add_batched_transition(batched_env_ret, batched_policy_resp, truncations)

        # log rollout stats
        env_stats = self.env_group.get_env_stats(reset=True)
        logger.info(f"{prefix}/stats:")
        log_metric(env_stats, step=self.epoch, prefix=prefix)

    def evaluate(self, obj_set: str, prefix: str = "") -> None:
        """Perform evaluation to collect experience from environments.

        Uses eval_env_list inside env_group if available, otherwise falls back
        to the training env_list.

        Args:
            obj_set: Object set identifier for environment reset options.
            prefix: Prefix for logging metrics.
        """
        self.env_group.apply_evaluate_cfg()
        batched_policy_resp = None
        max_rollout_steps = (
            self.config.train.max_eval_rollout_steps
            if self.config.train.max_eval_rollout_steps > 0
            else self.config.train.max_rollout_steps
        )
        with self.env_group.auto_reset(max_episode_steps=max_rollout_steps):
            for _ in range(max_rollout_steps + 1):
                if batched_policy_resp is None:
                    batched_env_ret, _ = self.env_group.reset(
                        options={"obj_set": obj_set},
                    )
                else:
                    batched_env_ret, _ = self.env_group.step(batched_policy_resp)
                batched_policy_resp = self.policy_group.rollout_batch(batched_env_ret)

        # log evaluation stats
        env_stats = self.env_group.get_env_stats(reset=True)
        logger.info(f"{prefix}/stats:")
        log_metric(env_stats, step=self.epoch, prefix=prefix)

        self.env_group.restore_evaluate_cfg()

    def update_dataset(self) -> None:
        """Update the dataset in the policy group from the buffer."""
        # prepare dataset
        data = self.buffer.sample(batch_size=self.config.train.batch_size)
        self.policy_group.update_dataset(data)

    def train(self) -> None:
        """Perform training on collected experience.

        Samples data from the buffer, updates the dataset, and trains
        the policy. Logs training info metrics.

        Raises:
            ValueError: If buffer size is smaller than batch size when
                sampling without replacement.
        """

        training_info = self.policy_group.train()

        # Ensure the returned training_info is a dict. If it's a list of dicts, aggregate by mean.
        if isinstance(training_info, list) and len(training_info):
            assert all(isinstance(x, dict) for x in training_info)

            def mean_leaf_fn(vals):
                vals = list(vals)
                return sum(vals) / len(vals) if vals else 0.0

            training_info = tree.map_structure(mean_leaf_fn, *training_info)

        if training_info is not None:
            log_metric(training_info, step=self.epoch, prefix="train")

    def run(self) -> None:
        """Run the main training loop.

        Executes the training loop for the configured number of epochs,
        performing rollout, training, and periodic evaluation.
        """
        for self.epoch in self.iter_epochs(num_epochs=self.config.train.max_epochs):
            self._rollout(obj_set="train", prefix="rollout")
            self._update_dataset()
            self._train()

            if self.config.train.eval_interval > 0 and self.epoch % self.config.train.save_interval == 0:
                ckpt_path = f"{self.config.train.save_dir}/epoch_{self.epoch}.pt"
                self.policy_group.save_checkpoint(path=ckpt_path)
            
            # sync weights after training and save checkpoint    
            self._sync_weights()

            if self.config.train.eval_interval > 0 and self.epoch % self.config.train.eval_interval == 0:
                logger.info(f"Evaluating at epoch {self.epoch}")
                self._evaluate(obj_set="train", prefix="eval")
                self._evaluate(obj_set="test", prefix="eval_ood")

            if InternalFlag.DEBUG:
                self.print_timing_summary()

        logger.info("Evaluating after final epoch...")
        self._evaluate(obj_set="train", prefix="eval")
        self._evaluate(obj_set="test", prefix="eval_ood")

        ckpt_path = f"{self.config.train.save_dir}/epoch_last.pt"
        self.policy_group.save_checkpoint(path=ckpt_path)

        logger.info("Done.")
