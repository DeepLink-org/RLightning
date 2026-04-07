"""Base policy module.

This module provides the abstract base class for all policies,
defining the interface for training, evaluation, and rollout operations.
"""

import asyncio
import os
import random
import threading
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from rlightning.types import EnvRet, PolicyResponse
from rlightning.utils.config import PolicyConfig, TrainConfig
from rlightning.utils.distributed.comm_context import CommContext
from rlightning.utils.distributed.group_initializer import ParallelMode
from rlightning.utils.logger import get_logger
from rlightning.utils.profiler import profiler
from rlightning.utils.ray import RayActorMixin
from rlightning.utils.utils import InternalFlag, to_numpy
from rlightning.weights.weight_buffer_mixin import WeightBufferMixin

logger = get_logger(__name__)


try:
    from enum import StrEnum
except ImportError:
    warnings.warn("no StrEnum in enum can be import, please ensure your python version is >= 3.11")
    from enum import Enum

    class StrEnum(str, Enum):
        pass


def set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """Set a dotted-path attribute on an object."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def infer_train_dp_world_size() -> int:
    """Infer TRAIN_DATA_PARALLEL world size, falling back safely to 1."""
    try:
        return CommContext().get_world_size(ParallelMode.TRAIN_DATA_PARALLEL)
    except Exception:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1


def clone_checkpoint_value(value: Any) -> Any:
    """Recursively clone checkpoint payloads into CPU-owned tensors."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()

    if isinstance(value, dict):
        return value.__class__((key, clone_checkpoint_value(child)) for key, child in value.items())

    if isinstance(value, list):
        return [clone_checkpoint_value(child) for child in value]

    if isinstance(value, tuple):
        return tuple(clone_checkpoint_value(child) for child in value)

    return value


class PolicyRole(StrEnum):
    """Policy role enumeration."""

    TRAIN = "train"
    EVAL = "eval"


class BasePolicy(nn.Module, RayActorMixin, WeightBufferMixin, ABC):
    """Abstract base class for reinforcement learning policies.

    This class provides the common interface for all RL policies,
    including methods for initialization, rollout, training, and
    weight management.

    """

    def __init__(
        self,
        config: PolicyConfig,
        role_type: PolicyRole,
    ) -> None:
        """Initialize the base policy.

        Args:
            config: Policy configuration object.
            role_type: Policy role (TRAIN or EVAL).
        """
        super().__init__()
        self.ready = False
        self.config = config
        self.device: str = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

        self.role_type = role_type
        self.model_list: List[Tuple[str, nn.Module]] = []

        self.is_init: bool = False

        # for eval policy
        self.rollout_mode: str = config.rollout_mode
        self._idle_as_infer: Optional[threading.Event] = None
        # for weight update
        self._weight_update_done: Optional[threading.Event] = None
        self._update_weights_signal: Optional[threading.Event] = None

        # for train policy
        self.dataset: Optional[Any] = None
        self.world_size: int = 1

        self.timing_raw: Dict[str, Dict[str, Any]] = {}

        # Initialize weight buffer mixin
        self.__init_weight_buffer_mixin__(config.weight_buffer.buffer_strategy)

        self._sanity_check()

    def _sanity_check(self) -> None:
        """Validate policy configuration.

        Raises:
            ValueError: If role_type is invalid.
        """
        if self.role_type not in [PolicyRole.TRAIN, PolicyRole.EVAL]:
            raise ValueError(f"Invalid role_type: {self.role_type}, must be one of {list(PolicyRole)}")

    def _find_model(self) -> None:
        """Find all trainable modules registered in the policy.

        Only modules with trainable parameters are added to ``model_list``.

        Raises:
            ValueError: If a registered model is not an ``nn.Module``.
        """

        if len(self.model_list) > 0:
            # check if all the models are nn.Module
            for module_name, model in self.model_list:
                if not isinstance(model, nn.Module):
                    raise ValueError(f"Invalid model: {module_name}, must be one of nn.Module")
        else:
            for attr_name, module in self._modules.items():
                try:
                    has_trainable_params = any(p.requires_grad for p in module.parameters())
                except Exception:
                    has_trainable_params = False
                    logger.exception(f"Raised an exception when checking if {attr_name} has trainable params")
                if has_trainable_params:
                    self.model_list.append((attr_name, module))

    @abstractmethod
    def construct_network(self, env_meta: Any, *args: Any, **kwargs: Any) -> None:
        """Construct the neural network architecture.

        Args:
            env_meta: Environment metadata for network configuration.
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def setup_optimizer(self, optim_cfg: Any) -> None:
        """Set up the optimizer for training.

        Args:
            optim_cfg: Optimizer configuration.
        """
        raise NotImplementedError

    def init_eval(self, eval_config: Optional[Any] = None, env_meta: Optional[Any] = None) -> None:
        """Initialize the policy for evaluation.

        Args:
            eval_config: Evaluation configuration.
            env_meta: Environment metadata.
        """
        self.eval_config = eval_config

        self.construct_network(env_meta=env_meta)
        self._find_model()
        for _, model in self.model_list:
            model.eval()

        if self.rollout_mode == "sync":
            self._idle_as_infer = threading.Event()
            self._idle_as_infer.set()
        elif self.rollout_mode == "async":
            self._loop = asyncio.get_event_loop()
            # admission control gate for new requests
            self._accept_new_requests = asyncio.Event()
            self._accept_new_requests.set()
            # track in-flight requests and provide a condition to wait for drain
            self.num_requests = 0
            self._num_requests_lock = threading.Lock()
            self._inflight_zero_cv = threading.Condition(self._num_requests_lock)
        else:
            raise ValueError(f"Invalid rollout mode: {self.rollout_mode}")

        if self.weight_buffer_strategy != "None":
            self._update_weights_signal = threading.Event()
            self._update_weights_signal.clear()
            self._weight_update_done = threading.Event()
            self._weight_update_done.set()  # initially "done" — no update in progress
            # always try to update weights for eval policy
            updater_thread = threading.Thread(target=self.update_weights, daemon=True)
            updater_thread.start()
        self.is_init = True

    def init_train(self, train_config: TrainConfig, env_meta: Optional[Any] = None) -> None:
        """Initialize the policy for training.

        Sets up the network, finds trainable models, wraps with DDP if needed,
        and initializes the optimizer.

        Args:
            train_config: Training configuration.
            env_meta: Environment metadata.
        """
        self.construct_network(env_meta=env_meta, backend="transformers")

        self._find_model()
        for _, model in self.model_list:
            model.train()

        parallel_mode = train_config.parallel
        self.world_size = infer_train_dp_world_size()
        if parallel_mode is None and self.world_size > 1:
            parallel_mode = "ddp"
        if parallel_mode is not None:
            if parallel_mode == "ddp":
                process_group = CommContext().get_group(ParallelMode.TRAIN_DATA_PARALLEL)
                # Barrier to ensure all ranks are ready before DDP initialization
                if process_group is not None:
                    torch.distributed.barrier(group=process_group)
                logic_gpu_id = CommContext().get_local_rank(ParallelMode.INTRA_NODE) % torch.cuda.device_count()
                wrapped_any = False
                if hasattr(self, "wrap_with_ddp"):
                    self.wrap_with_ddp(logic_gpu_id, process_group)
                    wrapped_any = True
                else:
                    for name, model in self.model_list:
                        if isinstance(model, nn.Module) and not isinstance(model, DDP):
                            ddp_mod = DDP(
                                model.cuda(),
                                device_ids=[logic_gpu_id],
                                process_group=process_group,
                                find_unused_parameters=True,
                            )
                            set_nested_attr(self, name, ddp_mod)
                            wrapped_any = True
                            logger.debug(f"wrap model with DDP: {name}")
                if not wrapped_any:
                    logger.warning("DDP set but no trainable nn.Module members on self to wrap.")
            else:
                raise ValueError(f"Unsupported parallel mode: {parallel_mode}")

        # optimizer
        self.setup_optimizer(self.config.optim_cfg)
        self.is_init = True

    def is_initialized(self) -> bool:
        """Return True if the policy has been initialized."""
        return self.is_init

    def notify_update_weights(self) -> None:
        """Signal that new weights are available for update."""
        # Mark as not-done before raising the signal so callers that
        # immediately call wait_for_weight_update_done() never miss the update.
        if self._weight_update_done is not None:
            self._weight_update_done.clear()
        self._update_weights_signal.set()

    def wait_for_weight_update_done(self, timeout: Optional[float] = None) -> bool:
        """Block until the background weight-update thread finishes its latest cycle.

        Args:
            timeout: Maximum seconds to wait. ``None`` means wait indefinitely.

        Returns:
            ``True`` if the update completed within *timeout*, ``False`` if it
            timed out.  Always returns ``True`` when there is no background
            update thread (``weight_buffer_strategy == "None"``).
        """
        if self._weight_update_done is None:
            return True
        return self._weight_update_done.wait(timeout=timeout)

    def get_param_fingerprint(self) -> Dict[str, float]:
        """Return per-parameter L2 norms as a lightweight weight fingerprint.

        Used by :meth:`PolicyGroup.verify_eval_weight_consistency` to detect
        weight-sync failures without shipping full tensors across processes.

        Returns:
            Dict mapping ``"<module_name>.<param_name>"`` to the float32 L2
            norm of each parameter tensor.
        """
        fingerprint: Dict[str, float] = {}
        for module_name, model in self.model_list:
            actual_model = model.module if isinstance(model, DDP) else model
            for param_name, param in actual_model.named_parameters():
                key = f"{module_name}.{param_name}"
                # In offload mode we may have freed CUDA storages (see WeightBufferMixin.offload_model_param_and_grad),
                # leaving parameters as CUDA tensors with zero storage. Fall back to the CPU backup when available.
                if param.data.storage().size() == 0:
                    cpu_backup = getattr(self, "cpu_param_backup", None)
                    # cpu_param_backup stores keys from self.model.named_parameters() without module prefix.
                    if isinstance(cpu_backup, dict) and module_name == "model" and param_name in cpu_backup:
                        cpu_tensor, _ = cpu_backup[param_name]
                        fingerprint[key] = cpu_tensor.detach().float().norm().item()
                    else:
                        # No backing storage available; report 0.0 rather than crashing.
                        fingerprint[key] = 0.0
                        logger.warning(
                            f"param {key} has no backing storage available; report 0.0 rather than crashing."
                        )
                else:
                    fingerprint[key] = param.detach().float().norm().item()
        return fingerprint

    def _pre_update_weights_hook(self) -> None:
        """Pre-update hook to wait for idle state before updating weights."""
        self._update_weights_signal.wait()
        if self.rollout_mode == "sync":
            # wait until policy is idle
            self._idle_as_infer.wait()
            self._idle_as_infer.clear()
        elif self.rollout_mode == "async":
            # stop accepting new requests at the very beginning of the update window
            self._loop.call_soon_threadsafe(self._accept_new_requests.clear)
            # Wait until there are no in-flight requests
            with self._num_requests_lock:
                while self.num_requests > 0:
                    self._inflight_zero_cv.wait()

    def _post_update_weights_hook(self) -> None:
        """Post-update hook to restore rollout capability."""
        if self.rollout_mode == "sync":
            self._idle_as_infer.set()
        elif self.rollout_mode == "async":
            self._loop.call_soon_threadsafe(self._accept_new_requests.set)
        self._update_weights_signal.clear()
        if self._weight_update_done is not None:
            self._weight_update_done.set()

    def update_weights(self) -> None:
        """Continuously update weights from weight_buffer.

        Runs in a daemon thread, waiting for update signals and
        applying new weights when available.
        """
        while True:
            self._pre_update_weights_hook()
            try:
                with profiler.timer("update_weights", self.timing_raw, level="debug", enable=InternalFlag.DEBUG):
                    self.update_weights_from_buffer()
            except Exception as exc:
                logger.exception("update_weights failed: %s", exc)
            finally:
                self._post_update_weights_hook()

    @abstractmethod
    @torch.inference_mode()
    def rollout_step(self, env_ret: EnvRet, **kwargs: Any) -> PolicyResponse:
        """Run a single policy step for the given environment return."""
        raise NotImplementedError

    @torch.inference_mode()
    def _rollout(self, env_ret: EnvRet, **kwargs: Any) -> PolicyResponse:
        """Synchronous rollout implementation (previous behavior)."""
        env_ret = self._pre_rollout_hook(env_ret)
        try:
            policy_resp = self.rollout_step(env_ret, **kwargs)
        except Exception as exc:
            logger.exception("rollout failed: %s", exc)
            raise exc
        return self._post_rollout_hook(policy_resp)

    @torch.inference_mode()
    async def _rollout_async(self, env_ret: EnvRet, **kwargs: Any) -> PolicyResponse:
        """Asynchronous rollout implementation for concurrent requests."""
        await self._pre_rollout_hook_async(env_ret)
        try:
            policy_resp = await self.rollout_step(env_ret, **kwargs)
        except Exception as exc:
            logger.exception("rollout failed: %s", exc)
            raise exc
        return self._post_rollout_hook_async(policy_resp)

    def _pre_rollout_hook(self, env_ret: EnvRet) -> EnvRet:
        """Prepare for sync rollout, tracking latency and idleness."""
        if InternalFlag.DEBUG:
            # compute env -> policy transfer time and record in timing_raw
            t_env_to_policy_s = env_ret.compute_sent_latency()
            profiler.record_timing("transition_env_to_policy", t_env_to_policy_s, self.timing_raw, level="debug")
        self._idle_as_infer.wait()
        self._idle_as_infer.clear()

        # Defensive device guard: if the model is on the wrong device (e.g. still
        # on CPU after an offload/reload race), move it back before inference.
        try:
            actual_device = next(self.parameters()).device
            if actual_device != torch.device(self.device):
                logger.warning(
                    "Model on %s but expected %s – moving model to correct device before rollout.",
                    actual_device,
                    self.device,
                )
                self.model.to(self.device)
                torch.cuda.synchronize()
        except StopIteration:
            pass

        return env_ret.cuda()

    async def _pre_rollout_hook_async(self, env_ret: EnvRet) -> None:
        """Prepare for async rollout, tracking latency and request counters."""
        if InternalFlag.DEBUG:
            # compute env -> policy transfer time and record in timing_raw
            t_env_to_policy_s = env_ret.compute_sent_latency()
            profiler.record_timing("transition_env_to_policy", t_env_to_policy_s, self.timing_raw, level="debug")
        await self._accept_new_requests.wait()
        with self._num_requests_lock:
            self.num_requests += 1

    def _post_rollout_hook(self, policy_resp: PolicyResponse) -> PolicyResponse:
        """Finalize sync rollout and normalize response device."""
        self._idle_as_infer.set()

        if InternalFlag.REMOTE_ENV or InternalFlag.REMOTE_STORAGE:
            policy_resp = policy_resp.numpy()

        policy_resp.mark_policy_sent()
        return policy_resp

    def _post_rollout_hook_async(self, policy_resp: PolicyResponse) -> PolicyResponse:
        """Finalize async rollout and normalize response device."""
        with self._num_requests_lock:
            self.num_requests -= 1
            if self.num_requests == 0:
                self._inflight_zero_cv.notify_all()

        if InternalFlag.REMOTE_ENV or InternalFlag.REMOTE_STORAGE:
            policy_resp = policy_resp.cpu()

        policy_resp.mark_policy_sent()
        return policy_resp

    @abstractmethod
    @torch.inference_mode()
    def postprocess(self, env_ret: Optional[EnvRet] = None, policy_resp: Optional[PolicyResponse] = None) -> Any:
        """Post-process environment and policy outputs."""

    @torch.inference_mode()
    def _postprocess(self, env_ret: Optional[EnvRet] = None, policy_resp: Optional[PolicyResponse] = None) -> Any:
        """Synchronous postprocess implementation (previous behavior)."""
        env_ret, policy_resp = self._pre_postprocess_hook(env_ret, policy_resp)
        try:
            result = self.postprocess(env_ret, policy_resp)
        except Exception as exc:
            logger.exception("postprocess failed: %s", exc)
            raise exc
        return self._post_postprocess_hook(result)

    def _pre_postprocess_hook(self, env_ret: Optional[EnvRet], policy_resp: Optional[PolicyResponse]) -> None:
        """Prepare for post-processing by acquiring idle state."""
        self._idle_as_infer.wait()
        self._idle_as_infer.clear()

        env_ret = env_ret.cuda() if env_ret is not None else None
        policy_resp = policy_resp.cuda() if policy_resp is not None else None

        return env_ret, policy_resp

    def _post_postprocess_hook(self, result: Any) -> Any:
        """Finalize post-processing and normalize response device."""

        if InternalFlag.REMOTE_ENV or InternalFlag.REMOTE_STORAGE:

            def _to_numpy(data: Any) -> Any:
                """Move nested results to CPU for remote-friendly return values."""
                if isinstance(data, (EnvRet, PolicyResponse)):
                    return data.numpy()
                else:
                    return to_numpy(data)

            if isinstance(result, (tuple, list)):
                result = type(result)(_to_numpy(r) for r in result)
            else:
                result = _to_numpy(result)

        self._idle_as_infer.set()

        return result

    @abstractmethod
    def update_dataset(self, data: TensorDict) -> None:
        """Update internal dataset from sampled buffer data."""
        raise NotImplementedError

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Run a training step for the policy."""
        raise NotImplementedError

    def check_idle(self) -> None:
        """Check if the policy is idle.

        Called by policy_group to determine if the actor is available
        for new requests.

        Raises:
            AssertionError: If called with async rollout mode.
        """
        assert self.rollout_mode == "sync", "check_idle is only supported for sync rollout mode"
        if self.role_type == PolicyRole.EVAL:
            self._idle_as_infer.wait()
        elif self.role_type == PolicyRole.TRAIN:
            pass

    def get_num_requests(self) -> int:
        """Get current number of in-flight requests.

        Returns:
            Number of in-flight requests for async rollout routing.
        """
        return self.num_requests if self.num_requests is not None else 0

    @abstractmethod
    def get_trainable_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return a dict of module state dicts."""

        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], *args: Any, **kwargs: Any) -> None:
        """Load trainable parameters from state_dict."""

        raise NotImplementedError

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint of the policy. User can override this method in subclass if needed.

        Args:
            path (str): The path to save the checkpoint.
        """
        ckpt_folder = Path(path).parent
        os.makedirs(ckpt_folder, exist_ok=True)

        model_was_offloaded = getattr(self, "_model_params_offloaded", False)
        if model_was_offloaded:
            self.reload_model_param_and_grad(load_grad=False)

        try:
            state: Dict[str, Dict] = {}
            for name, model in self.model_list:
                module = model.module if isinstance(model, DDP) else model
                state[name] = clone_checkpoint_value(module.state_dict())

            torch.save(state, path)
        finally:
            if model_was_offloaded:
                self.offload_model_param_and_grad(offload_grad=False)

    def reset_training_state(
        self, train_config: TrainConfig, env_meta: Optional[Any] = None, seed: Optional[int] = None
    ) -> None:
        """Reset model + optimizer to initial state after warm_up."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # avoid duplicate model registrations
        self.model_list = []

        self._rebuild_after_warmup(train_config, env_meta)

    def _rebuild_after_warmup(self, train_config: TrainConfig, env_meta: Optional[Any] = None) -> None:
        """Internal rebuild hook for warm_up reset.

        Subclasses typically should not override this private hook; the default
        behavior is to re-run training initialization.
        """
        self.dataset = None
        if hasattr(self, "optimizer"):
            self.optimizer = None
        if hasattr(self, "optimizer_steps"):
            self.optimizer_steps = 0
        if hasattr(self, "_setup_sampling_params"):
            self._setup_sampling_params()
        for name in list(self._modules.keys()):
            if name.startswith("rsl_rl/"):
                self._modules.pop(name, None)
        self.init_train(train_config, env_meta=env_meta)

    def print_timing_summary(self, reset: bool = False) -> None:
        """Print timing summary for profiling.

        Args:
            reset: If True, reset timing statistics after printing.
        """
        role = getattr(self, "role_type", "unknown")
        logger.debug(f"Policy ({role}) timing:")
        # iterate over a snapshot to avoid concurrent modification during iteration
        for name, stats in dict(self.timing_raw).items():
            logger.debug(f"{name:28} count={stats['count']:<3} " f"total={stats['total']:.6f}s avg={stats['avg']:.6f}s")
        if reset:
            self.timing_raw = {}
