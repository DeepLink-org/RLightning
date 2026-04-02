import gc

import ray
import torch
from torch.distributed.utils import _alloc_storage, _free_storage
from torch.nn.parallel import DistributedDataParallel as DDP

from rlightning.utils.common import MultiprocessingSerializer
from rlightning.utils.distributed.comm_context import CommContext
from rlightning.utils.distributed.group_initializer import ParallelMode
from rlightning.utils.logger import get_logger
from rlightning.utils.profiler import profiler
from rlightning.utils.utils import InternalFlag
from rlightning.weights import WeightTransferManager
from rlightning.weights.utils import (
    build_weight_buffer,
    is_numpy_state_dict,
    numpy_state_dict_to_tensor,
)

logger = get_logger(__name__)


class WeightBufferMixin:
    """
    Mixin class for weight buffer management functionality.
    This mixin provides methods for initializing, updating, and managing weight buffers
    for policy weight transfer between train and eval policies.
    """

    def __init_weight_buffer_mixin__(self, buffer_strategy: str):
        """
        Initialize weight buffer related attributes.
        This method should be called in the __init__ method of the class that uses this mixin.
        """
        # for weight transfer
        self.weight_buffer = None
        self.weight_buffer_strategy = buffer_strategy
        self.weight_transfer_manager = WeightTransferManager(buffer_strategy=buffer_strategy)
        self.weight_transfer_times = []
        self.weight_update_times = []
        self._update_weights_signal = None

        # for offload model param and grad
        self.cpu_param_backup = {}
        self._model_params_offloaded = False

    def init_weight_buffer(self, shared_weight_buffer=None):
        """Initialize the weight buffer."""
        # Check role_type value to avoid circular import
        assert (hasattr(self.role_type, "value") and self.role_type.value == "eval") or str(
            self.role_type
        ) == "eval", "Only eval policy can init weight buffer"
        if shared_weight_buffer is not None:
            self.weight_buffer = shared_weight_buffer
        else:
            self.weight_buffer = build_weight_buffer(self.config.weight_buffer.type, self.config.weight_buffer)

    def get_weight_buffer(self):
        """Get the weight buffer instance."""
        return self.weight_buffer

    def update_weights_from_buffer(self):
        """Update weights from the weight buffer."""
        if isinstance(self.weight_buffer, ray.actor.ActorHandle):
            state_dict = ray.get(self.weight_buffer.get_state_dict.remote())
        else:
            state_dict = self.weight_buffer.get_state_dict()
        if is_numpy_state_dict(state_dict):
            state_dict = numpy_state_dict_to_tensor(state_dict)
        # check if load_state_dict is implemented
        self.load_state_dict(state_dict)
        if InternalFlag.DEBUG:
            # record gpu memory and time
            self.weight_transfer_manager.record_gpu_memory("update_weights")

    def update_weights(self):
        """
        Update the weights from weight_buffer.
        This method runs in a daemon thread and continuously waits for update signals.
        """
        while True:
            self._pre_update_weights_hook()
            try:
                with profiler.timer(
                    "update_weights",
                    self.timing_raw,
                    level="debug",
                    enable=InternalFlag.DEBUG,
                ):
                    self.update_weights_from_buffer()
            except Exception:
                logger.exception("update_weights failed")
            finally:
                self._post_update_weights_hook()

    @profiler.timer_wrap(level="debug")
    def send_weights(self, eval_policys, shared_weight_buffer=None):
        """
        Send the weights to the eval policy.
        """

        # check if get_trainable_parameters is implemented
        state_dict = self.get_trainable_parameters()
        if shared_weight_buffer is not None:
            # Now shared weight buffer only support CPUWeightBuffer
            self.weight_transfer_manager.send_weights_cpu(state_dict, shared_weight_buffer)
        if len(eval_policys) > 0:
            self.weight_transfer_manager.send_weights(state_dict, eval_policys)

    def send_weights_ipc(self):
        """
        Send the weights to the eval policy using IPC.
        """
        state_dict = self.get_trainable_parameters()
        serialized_state_dict = MultiprocessingSerializer.serialize(state_dict, output_str=True)
        return serialized_state_dict

    def recv_weights(self, metadata_by_dtype):
        """
        Receive the weights from the train policy.
        Only call this method from train policy, in weight_transfer_manager.send_weights()
        """
        self.weight_transfer_manager.recv_weights(self.weight_buffer, metadata_by_dtype)

    def recv_weights_ipc(self, data):
        """
        Receive the weights from the eval policy using IPC.
        """
        deserialized_state_dict = MultiprocessingSerializer.deserialize(data)
        self.load_state_dict(deserialized_state_dict)
        del deserialized_state_dict
        self.clear_memory()

    def recv_weights_local(self, state_dict):
        """
        This method is used to receive weights locally without using ray in running on a
        single process.
        """
        self.weight_transfer_manager.recv_weights_local(self.weight_buffer, state_dict)

    def sync_weights_layer_by_layers(self, module_name, name, dtype, shape, is_last: bool = False):
        """
        Sync the weight from the train policy layer by layer.
        """
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        weight_transfer_group = CommContext().get_group(ParallelMode.WEIGHT_TRANSFER)
        # The sender (train) is always at position 0 in the WEIGHT_TRANSFER ranks list.
        src_rank = CommContext().get_ranks_in_group(ParallelMode.WEIGHT_TRANSFER)[0]

        torch.distributed.broadcast(weight, src=src_rank, group=weight_transfer_group)

        state_dict = {module_name: {name: weight}}
        self.load_state_dict_layer_by_layer(state_dict, is_last)  # not implemented

    def offload_model(self):
        """
        offload model to cpu
        """
        self.model.to("cpu", non_blocking=True)
        self.clear_memory(sync=True)
        profiler.log_gpu_memory_usage("offload_model")

    def reload_model(self):
        """
        load model to device
        """
        self.model.to(self.device, non_blocking=True)
        self.clear_memory(sync=True)
        profiler.log_gpu_memory_usage("reload_model")

    def offload_model_param_and_grad(self, offload_grad=False):
        """
        offload model param and grad to cpu
        """

        assert self.model is not None, "Now only support model is self.model"
        actual_model = self.model.module if isinstance(self.model, DDP) else self.model
        for name, param in actual_model.named_parameters():
            if param.data.storage().size() > 0:
                self.cpu_param_backup[name] = (param.data.cpu(), param.data.size())
                _free_storage(param.data)
            if offload_grad and param.grad is not None:
                param.grad = param.grad.to("cpu", non_blocking=True)
        self._model_params_offloaded = True
        self.clear_memory(sync=True)
        profiler.log_gpu_memory_usage("offload_model_param_and_grad")

    def reload_model_param_and_grad(self, load_grad=False):
        """
        load model param and grad to device
        """
        assert self.model is not None, "Now only support model is self.model"
        actual_model = self.model.module if isinstance(self.model, DDP) else self.model
        for name, param in actual_model.named_parameters():
            if name in self.cpu_param_backup and param.data.storage().size() == 0:
                cpu_tensor, size = self.cpu_param_backup[name]
                _alloc_storage(param.data, size)
                param.data.copy_(cpu_tensor)

            if load_grad and param.grad is not None:
                param.grad = param.grad.to(self.device, non_blocking=True)
        self._model_params_offloaded = False
        self.clear_memory(sync=True)
        profiler.log_gpu_memory_usage("reload_model_param_and_grad")

    def offload_optimizer(self):
        """
        offload optimizer to cpu
        """
        if not self.optimizer.state:
            return

        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value.to("cpu", non_blocking=True)
        self.clear_memory(sync=True)
        profiler.log_gpu_memory_usage("offload_optimizer")

    def reload_optimizer(self):
        """
        load optimizer to device
        """
        if not self.optimizer.state:
            return

        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value.to(self.device, non_blocking=True)
        self.clear_memory(sync=True)
        profiler.log_gpu_memory_usage("reload_optimizer")

    def clear_memory(self, sync=False):
        """
        Clear the memory.
        """
        if sync:
            torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
