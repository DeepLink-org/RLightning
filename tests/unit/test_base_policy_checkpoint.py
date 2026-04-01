"""Focused tests for BasePolicy checkpoint serialization behavior."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
from torch.distributed.utils import _free_storage

from rlightning.policy.base_policy import BasePolicy, PolicyRole, clone_checkpoint_value
from rlightning.types import EnvRet, PolicyResponse
from rlightning.utils.config import PolicyConfig


class _CheckpointPolicy(BasePolicy):
    """Concrete policy used to exercise checkpoint save paths."""

    def construct_network(self, env_meta: Any = None, *args: Any, **kwargs: Any) -> None:
        self.linear = nn.Linear(2, 2)

    def setup_optimizer(self, optim_cfg: Any) -> None:
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def rollout_step(self, env_ret: EnvRet, **kwargs: Any) -> PolicyResponse:
        return PolicyResponse(env_id=env_ret.env_id, action=torch.tensor(1))

    def postprocess(self, env_ret: EnvRet | None = None, policy_resp: PolicyResponse | None = None) -> Any:
        return env_ret, policy_resp

    def update_dataset(self, data: Any) -> None:
        self.dataset = data

    def train(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def get_trainable_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {name: module.state_dict() for name, module in self.model_list}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], *args: Any, **kwargs: Any) -> None:
        self.loaded_state = state_dict


def _make_policy() -> _CheckpointPolicy:
    config = PolicyConfig.from_dict({"type": "CheckpointPolicy", "rollout_mode": "sync"})
    policy = _CheckpointPolicy(config, PolicyRole.TRAIN)
    policy.construct_network()
    policy._find_model()
    policy.model = policy.linear
    return policy


def test_clone_checkpoint_value_produces_loadable_cpu_tensors(tmp_path):
    """clone_checkpoint_value should detach tensor payloads into standalone CPU storages."""
    source = {
        "view": torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, :2],
        "nested": (torch.arange(4, dtype=torch.bfloat16).reshape(2, 2),),
    }

    cloned = clone_checkpoint_value(source)

    assert cloned["view"].device.type == "cpu"
    assert cloned["nested"][0].device.type == "cpu"
    assert cloned["view"].data_ptr() != source["view"].data_ptr()

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(cloned, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    torch.testing.assert_close(loaded["view"], source["view"])
    torch.testing.assert_close(loaded["nested"][0], source["nested"][0])


def test_save_checkpoint_round_trips_offloaded_parameters(tmp_path, monkeypatch):
    """save_checkpoint should temporarily reload zero-storage params and restore offload state."""
    policy = _make_policy()
    original_state = {name: tensor.detach().cpu().clone() for name, tensor in policy.linear.state_dict().items()}
    monkeypatch.setattr(policy, "clear_memory", lambda sync=False: None)
    monkeypatch.setattr(
        "rlightning.weights.weight_buffer_mixin.profiler.log_gpu_memory_usage",
        lambda *args, **kwargs: None,
    )

    for name, param in policy.linear.named_parameters():
        policy.cpu_param_backup[name] = (param.data.detach().cpu().clone(), param.data.size())
        _free_storage(param.data)

    for param in policy.linear.parameters():
        assert param.data.storage().size() == 0

    checkpoint_path = tmp_path / "epoch_1" / "model.pt"
    policy.save_checkpoint(checkpoint_path)

    assert checkpoint_path.exists()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "linear" in state
    for name, tensor in original_state.items():
        torch.testing.assert_close(state["linear"][name], tensor)

    for param in policy.linear.parameters():
        assert param.data.storage().size() == 0
