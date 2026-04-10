"""Asynchronous reinforcement learning engine module.

This module implements the AsyncRLEngine for asynchronous training where
rollout, training, and weight updates run in separate threads.
"""

import threading
import time

from rlightning.buffer.base_buffer import DataBuffer
from rlightning.env.env_group import EnvGroup
from rlightning.policy import PolicyGroup
from rlightning.utils.config import MainConfig
from rlightning.utils.logger import get_logger, log_metric
from rlightning.utils.profiler import profiler
from rlightning.utils.registry import ENGINE
from rlightning.utils.utils import InternalFlag

from .base_engine import BaseEngine

logger = get_logger(__name__)


class AsyncCoordinator:
    """A coordinator to manage asynchronous tasks."""

    def __init__(self):
        self._done_event = threading.Event()

        self._ready_for_update_dataset_event = threading.Event()
        self._ready_for_sync_weights_event = threading.Event()
        self._weights_updated_event = threading.Event()
        self._dataset_ready_event = threading.Event()

    def start(self) -> None:
        """Start the coordinator."""
        self._done_event.clear()
        self._ready_for_update_dataset_event.set()
        self._ready_for_sync_weights_event.clear()
        self._weights_updated_event.set()
        self._dataset_ready_event.clear()

    def stop(self) -> None:
        """Stop the coordinator."""
        self._done_event.set()
        self._ready_for_update_dataset_event.set()
        self._ready_for_sync_weights_event.set()
        self._weights_updated_event.set()
        self._dataset_ready_event.set()

    def is_running(self) -> bool:
        """Check if the coordinator is still running."""
        return not self._done_event.is_set()

    def wait_for_dataset_ready(self) -> None:
        """Wait until dataset is ready for training."""
        self._dataset_ready_event.wait()
        self._dataset_ready_event.clear()

    def wait_for_weights_updated(self) -> None:
        """Wait until weights are updated for evaluation."""
        self._weights_updated_event.wait()
        self._weights_updated_event.clear()

    def wait_for_update_dataset(self) -> None:
        """Wait for signal to update dataset."""
        self._ready_for_update_dataset_event.wait()
        self._ready_for_update_dataset_event.clear()

    def wait_for_sync_weights(self) -> None:
        """Wait for signal to sync weights."""
        self._ready_for_sync_weights_event.wait()
        self._ready_for_sync_weights_event.clear()

    def notify_train_step_done(self) -> None:
        """Notify that the training step is done."""
        self._ready_for_update_dataset_event.set()
        self._ready_for_sync_weights_event.set()

    # ==
    def notify_dataset_ready(self) -> None:
        """Notify that dataset is ready for training."""
        self._dataset_ready_event.set()

    def notify_weight_update_step_done(self) -> None:
        """Notify that the weight update is done."""
        self._weights_updated_event.set()


@ENGINE.register("asyncrl")
class AsyncRLEngine(BaseEngine):
    """Asynchronous reinforcement learning engine.

    This engine implements asynchronous training where:
    - Rollout thread: Collects experience from environments.
    - Training thread: Updates policy using collected experience.
    - Weight update thread: Broadcasts updated weights to eval policies.

    """

    def __init__(
        self,
        config: MainConfig,
        env_group: EnvGroup,
        policy_group: PolicyGroup,
        buffer: DataBuffer,
    ) -> None:
        """Initialize the async RL engine.

        Args:
            config: Main configuration object.
            env_group: Environment group for managing multiple environments.
            policy_group: Policy group for managing train and eval policies.
            buffer: Data buffer for storing and sampling experience.
        """
        super().__init__(config, env_group, policy_group, buffer)

        self.coordinator = AsyncCoordinator()

        # init env group
        env_meta_list = self.env_group.init()
        self.env_meta_list = env_meta_list
        # init eval policy
        episode_meta = self.policy_group.init_eval(env_meta=env_meta_list[0])
        # init train policy
        self.policy_group.init_train(self.config.train, env_meta_list[0])
        # init replay buffer
        env_ids = self.env_group.env_ids
        self.buffer.init(env_meta_list, env_ids)

        if InternalFlag.DEBUG:
            self.warm_up()
        else:
            self.sync_weights()

    def warm_up(self):
        """init and dummy run for constructing RL dataflow
        Performs a dummy rollout and training iteration to ensure proper
        dataflow construction.
        """

        env_meta_list = self.env_meta_list

        batched_policy_resp = None
        for _ in range(11):
            if batched_policy_resp is None:
                batched_env_ret, _ = self.env_group.reset(seed=0)
            else:
                batched_env_ret, _ = self.env_group.step(batched_policy_resp)
            self.buffer.add_batched_data_async(batched_env_ret)

            batched_policy_resp = self.policy_group.rollout_batch(batched_env_ret)
            self.buffer.add_batched_data_async(batched_policy_resp)

        self.buffer.truncate_episodes(batched_policy_resp.ids())

        # dummy run train
        data = self.buffer.sample(batch_size=10)
        self.policy_group.update_dataset(data)
        self.policy_group.train()

        # reset
        if InternalFlag.DEBUG:
            self.print_timing_summary(reset=True)
        self.buffer.clear()
        # reset training state after dummy run
        seed = getattr(self.config.train, "seed", None)
        self.policy_group.reset_training_state(self.config.train, env_meta_list[0], seed=seed)
        self.sync_weights()

        logger.info("Warm up done, ready to run.")

    def rollout(self) -> None:
        """Perform rollout to collect experience from environments.

        Runs the rollout loop, collecting experience by stepping through
        environments and storing transitions in the buffer. Runs until
        the done_flag is set.
        """
        batched_policy_resp = None
        with self.env_group.auto_reset():
            while self.coordinator.is_running():
                if batched_policy_resp is None:
                    batched_env_ret, truncations = self.env_group.reset(seed=0)
                else:
                    self.env_group.step_async(batched_policy_resp)
                    batched_env_ret, truncations = self.env_group.collect_async()
                self.buffer.add_batched_data_async(batched_env_ret)
                batched_policy_resp = self.policy_group.rollout_batch(batched_env_ret)

                self.buffer.add_batched_data_async(batched_policy_resp, truncations)

    def evaluate(self) -> None:
        """Evaluate is not implemented for async engine."""
        raise NotImplementedError("Evaluate is not implemented for async engine.")

    def update_dataset(self) -> None:
        """Update dataset from buffer to train policy."""
        data = self.buffer.sample(batch_size=self.config.train.batch_size)
        self.policy_group.update_dataset(data)

    @profiler.timer_wrap(name="wait_for_data", level="info")
    def _pre_update_dataset_hook(self):
        """wait until enough data is collected in buffer"""
        if self.buffer.size() >= self.config.train.batch_size:
            return

        while self.coordinator.is_running():
            current_size = self.buffer.size()
            if current_size >= self.config.train.batch_size:
                return
            elif current_size == 0:
                time.sleep(0.1)
            else:
                time.sleep(0.01)

    def _update_dataset_loop(self):
        """Update dataset loop for async engine."""
        while self.coordinator.is_running():
            self.coordinator.wait_for_update_dataset()
            if not self.coordinator.is_running():
                break

            self._update_dataset()
            self.coordinator.notify_dataset_ready()

    def train(self) -> None:
        """Perform training on collected experience.

        Runs the training loop for the configured number of epochs,
        sampling from the buffer and updating the policy. Sets the
        new_weights_ready flag after each training step and done_flag
        when training completes.
        """
        train_info = self.policy_group.train()
        if train_info is not None and train_info:
            logger.info("train/info:")
            log_metric(train_info, step=self.epoch, prefix="train")

    def _train_loop(self) -> None:
        """Training loop with coordinator checks for async engine.

        Overrides the base training loop to include coordinator
        synchronization for asynchronous training.
        """
        try:
            self.coordinator.start()
            for self.epoch in self.iter_epochs(self.config.train.max_epochs):
                self.coordinator.wait_for_dataset_ready()
                self.coordinator.wait_for_weights_updated()
                self._train()
                if self.config.train.save_interval > 0 and self.epoch % self.config.train.save_interval == 0:
                    ckpt_path = f"{self.config.train.save_dir}/epoch_{self.epoch}.pt"
                    self.policy_group.save_checkpoint(path=ckpt_path)
                self.coordinator.notify_train_step_done()

        finally:
            self.coordinator.stop()

    def _sync_weights_loop(self):
        while self.coordinator.is_running():
            self.coordinator.wait_for_sync_weights()
            if not self.coordinator.is_running():
                break
            self._sync_weights()
            self.coordinator.notify_weight_update_step_done()

    def run(self) -> None:
        """Launch asynchronous training threads.

        Starts three worker threads (rollout, train, weight update) and
        waits for them to complete. Prints timing summary when finished.
        """
        workers = [
            threading.Thread(target=self._rollout),
            threading.Thread(target=self._update_dataset_loop),
            threading.Thread(target=self._train_loop),
            threading.Thread(target=self._sync_weights_loop),
        ]

        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
        except Exception as e:
            logger.error(f"Exception in worker threads: {e}")
            raise e

        if InternalFlag.DEBUG:
            self.print_timing_summary()

        ckpt_path = f"{self.config.train.save_dir}/epoch_last.pt"
        self.policy_group.save_checkpoint(path=ckpt_path)

        logger.info("Done.")
