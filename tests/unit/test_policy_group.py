"""Unit tests for PolicyGroup and transfer-info helpers."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List

import pytest
import torch

from rlightning.policy.base_policy import PolicyRole
from rlightning.policy.policy_group import (
    PolicyGroup,
    build_transfer_info_for_colocated,
    build_transfer_info_for_disaggregated,
)
from rlightning.types import BatchedData, EnvRet, PolicyResponse
from rlightning.utils.config import PolicyConfig, TrainConfig, WeightBufferConfig
from rlightning.utils.distributed.group_initializer import ParallelMode

@pytest.fixture(autouse=True)
def _reset_remote_flags(monkeypatch):
    """Default unit tests to local mode unless a test opts into remote mode."""
    monkeypatch.setenv("RLIGHTNING_REMOTE_TRAIN", "0")
    monkeypatch.setenv("RLIGHTNING_REMOTE_EVAL", "0")


class _DummyPolicy:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: Dict[str, List] = {
            "init_eval": [],
            "init_train": [],
            "rollout": [],
            "postprocess": [],
            "update_dataset": [],
            "train": [],
            "print_timing_summary": [],
            "notify_update_weights": 0,
            "reset_training_state": [],
            "send_weights": [],
            "save_checkpoint": [],
            "init_weight_buffer": [],
        }

    def init_eval(self, eval_config=None, env_meta=None):
        self.calls["init_eval"].append((eval_config, env_meta))
        return f"eval-{self.name}"

    def init_train(self, train_config=None, env_meta=None):
        self.calls["init_train"].append((train_config, env_meta))
        return f"train-{self.name}"

    def _rollout(self, env_ret: EnvRet):
        self.calls["rollout"].append(env_ret.env_id)
        return PolicyResponse(env_id=env_ret.env_id, action=torch.tensor(1))

    def _postprocess(self, *args: Any):
        self.calls["postprocess"].append(args)
        return (self.name, args)

    def update_dataset(self, data: Any):
        self.calls["update_dataset"].append(data)
        return f"update-{self.name}"

    def train(self, *args: Any, **kwargs: Any):
        self.calls["train"].append((args, kwargs))
        return {"policy": self.name}

    def print_timing_summary(self, reset: bool = False):
        self.calls["print_timing_summary"].append(reset)

    def notify_update_weights(self):
        self.calls["notify_update_weights"] += 1

    def reset_training_state(self, train_config, env_meta=None, seed=None):
        self.calls["reset_training_state"].append((train_config, env_meta, seed))

    def send_weights(self, receivers, shared_weight_buffer=None):
        self.calls["send_weights"].append((receivers, shared_weight_buffer))

    def save_checkpoint(self, save_dir):
        self.calls["save_checkpoint"].append(save_dir)

    def init_weight_buffer(self, shared_weight_buffer=None):
        self.calls["init_weight_buffer"].append(shared_weight_buffer)


class _FakeObjectRef:
    def __init__(self, value=None, hex_value: str | None = None) -> None:
        self.value = value
        self._hex = hex_value or hex(id(self))

    def hex(self) -> str:
        return self._hex


class _RemoteMethod:
    def __init__(self, fn, wrap_ref: bool = False):
        self.fn = fn
        self.wrap_ref = wrap_ref
        self.calls: List = []

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        result = self.fn(*args, **kwargs)
        if self.wrap_ref:
            return _FakeObjectRef(result)
        return result


class _AsyncPolicy:
    def __init__(self, name: str, num_requests: int) -> None:
        self.name = name
        self._num_requests = num_requests
        self.rollout_calls: List[str] = []
        self.get_num_requests = _RemoteMethod(self._get_num_requests)
        self._rollout_async = _RemoteMethod(self._rollout)

    def _get_num_requests(self):
        return self._num_requests

    def _rollout(self, env_ret: EnvRet):
        self.rollout_calls.append(env_ret.env_id)
        return f"resp-{self.name}-{env_ret.env_id}"


def _make_policy_cfg(
    *,
    rollout_mode: str = "sync",
    buffer_strategy: str = "Double",
    buffer_type: str = "WeightBuffer",
    is_colocated: bool = False,
) -> PolicyConfig:
    return PolicyConfig(
        type="DummyPolicy",
        rollout_mode=rollout_mode,
        is_colocated=is_colocated,
        weight_buffer=WeightBufferConfig(type=buffer_type, buffer_strategy=buffer_strategy),
    )


def test_policy_group_init_calls_eval_then_train():
    """init should call init_eval and init_train for all policies."""
    train_policy = _DummyPolicy("train")
    eval_policy = _DummyPolicy("eval")
    group = PolicyGroup([train_policy], [eval_policy], _make_policy_cfg())
    train_cfg = TrainConfig(max_epochs=1)

    group.init(train_cfg, env_meta="meta", eval_config="eval-cfg")

    assert eval_policy.calls["init_eval"] == [("eval-cfg", "meta")]
    assert train_policy.calls["init_train"] == [(train_cfg, "meta")]


def test_policy_group_rollout_batch_sync_uses_idle_deque():
    """rollout_batch should dispatch sync rollouts via idle deque."""
    eval_policy = _DummyPolicy("eval-1")
    group = PolicyGroup([], [eval_policy], _make_policy_cfg(rollout_mode="sync"))
    group._idle_deque = deque()

    env_rets = {
        "env-1": EnvRet(env_id="env-1", observation=0),
        "env-2": EnvRet(env_id="env-2", observation=1),
    }
    batched = BatchedData.from_dict(env_rets)

    result = group.rollout_batch(batched)

    assert result.ids() == ("env-1", "env-2")
    assert [resp.env_id for resp in result.values()] == ["env-1", "env-2"]
    assert eval_policy.calls["rollout"] == ["env-1", "env-2"]


def test_policy_group_rollout_batch_async_uses_router(monkeypatch):
    """rollout_batch should route async requests based on router assignments."""
    policies = [_AsyncPolicy("p0", num_requests=2), _AsyncPolicy("p1", num_requests=0)]
    group = PolicyGroup([], policies, _make_policy_cfg(rollout_mode="async"))

    assign_calls: Dict[str, List[int]] = {}

    def fake_assign(loads, n):
        assign_calls["loads"] = loads
        assign_calls["n"] = n
        return [1, 0]

    group.router.assign = fake_assign
    monkeypatch.setattr("rlightning.policy.policy_group.ray.get", lambda x: x)

    env_rets = {
        "env-1": EnvRet(env_id="env-1", observation=0),
        "env-2": EnvRet(env_id="env-2", observation=1),
    }
    batched = BatchedData.from_dict(env_rets)

    result = group.rollout_batch(batched)

    assert assign_calls["loads"] == [2, 0]
    assert assign_calls["n"] == 2
    assert policies[1].rollout_calls == ["env-1"]
    assert policies[0].rollout_calls == ["env-2"]
    assert result.values() == ("resp-p1-env-1", "resp-p0-env-2")


def test_policy_group_postprocess_sync_dispatches():
    """postprocess should dispatch to eval policies and preserve ids."""
    eval_policy = _DummyPolicy("eval-1")
    group = PolicyGroup([], [eval_policy], _make_policy_cfg())
    group._idle_deque = deque()

    env_rets = {
        "env-1": EnvRet(env_id="env-1", observation=0),
        "env-2": EnvRet(env_id="env-2", observation=1),
    }
    policy_resps = {
        "env-1": PolicyResponse(env_id="env-1", action=1),
        "env-2": PolicyResponse(env_id="env-2", action=2),
    }

    batched_env = BatchedData.from_dict(env_rets)
    batched_resp = BatchedData.from_dict(policy_resps)

    result = group.postprocess(batched_env, batched_resp)

    assert result.ids() == ("env-1", "env-2")
    assert len(result.values()) == 2
    assert eval_policy.calls["postprocess"]


def test_policy_group_postprocess_env_only():
    """postprocess should work with only batched_env_ret."""
    eval_policy = _DummyPolicy("eval-1")
    group = PolicyGroup([], [eval_policy], _make_policy_cfg())
    group._idle_deque = deque([eval_policy])

    env_rets = {"env-1": EnvRet(env_id="env-1", observation=0)}
    batched_env = BatchedData.from_dict(env_rets)

    result = group.postprocess(batched_env, None)

    assert result.ids() == ("env-1",)
    name, args = result.values()[0]
    assert name == "eval-1"
    assert len(args) == 1
    assert isinstance(args[0], EnvRet)


def test_policy_group_postprocess_policy_only():
    """postprocess should work with only batched_policy_resp."""
    eval_policy = _DummyPolicy("eval-1")
    group = PolicyGroup([], [eval_policy], _make_policy_cfg())
    group._idle_deque = deque([eval_policy])

    policy_resps = {"env-1": PolicyResponse(env_id="env-1", action=1)}
    batched_resp = BatchedData.from_dict(policy_resps)

    result = group.postprocess(None, batched_resp)

    assert result.ids() == ("env-1",)
    name, args = result.values()[0]
    assert name == "eval-1"
    assert len(args) == 1
    assert isinstance(args[0], PolicyResponse)


def test_policy_group_update_dataset_and_train():
    """update_dataset and train should route to all train policies."""
    train_policies = [_DummyPolicy("t1"), _DummyPolicy("t2")]
    group = PolicyGroup(train_policies, [], _make_policy_cfg())

    updates = group.update_dataset(["d1", "d2"])
    assert updates == ["update-t1", "update-t2"]
    assert train_policies[0].calls["update_dataset"] == ["d1"]

    train_info = group.train(123)
    assert train_info == {"policy": "t1"}


def test_policy_group_update_dataset_mismatched_length_raises():
    """update_dataset should assert when length mismatches train policies."""
    train_policies = [_DummyPolicy("t1")]
    group = PolicyGroup(train_policies, [], _make_policy_cfg())

    with pytest.raises(AssertionError):
        group.update_dataset([])


def test_policy_group_reset_training_state_calls_policies():
    """reset_training_state should call all train policies."""
    train_policies = [_DummyPolicy("t1"), _DummyPolicy("t2")]
    group = PolicyGroup(train_policies, [], _make_policy_cfg())
    train_cfg = TrainConfig(max_epochs=1)

    group.reset_training_state(train_cfg, env_meta="meta", seed=123)

    assert train_policies[0].calls["reset_training_state"] == [(train_cfg, "meta", 123)]
    assert train_policies[1].calls["reset_training_state"] == [(train_cfg, "meta", 123)]


def test_policy_group_reset_training_state_remote(monkeypatch):
    """reset_training_state should ray.get refs and call dist_barrier for remote refs."""
    train_policy = _DummyPolicy("t1")
    train_policy.dist_barrier = _RemoteMethod(lambda *args: "barrier", wrap_ref=True)
    group = PolicyGroup([train_policy], [], _make_policy_cfg())
    train_cfg = TrainConfig(max_epochs=1)

    def fake_submit(method, *args, _block=False, **kwargs):
        result = method(*args, **kwargs)
        if _block:
            return result
        return _FakeObjectRef(result)

    def fake_ray_get(obj):
        if isinstance(obj, list):
            return [fake_ray_get(o) for o in obj]
        if isinstance(obj, _FakeObjectRef):
            return obj.value
        return obj

    monkeypatch.setattr("rlightning.policy.policy_group.ray.ObjectRef", _FakeObjectRef, raising=False)
    monkeypatch.setattr("rlightning.policy.policy_group.ray.get", fake_ray_get)

    group._eval_task_submitter.submit = fake_submit
    group._train_task_submitter.submit = fake_submit
    group.reset_training_state(train_cfg, env_meta="meta", seed=321)

    assert train_policy.calls["reset_training_state"] == [(train_cfg, "meta", 321)]
    assert train_policy.dist_barrier.calls[0][0][0] == ParallelMode.TRAIN_DATA_PARALLEL


def test_policy_group_print_timing_summary_calls_policies():
    """print_timing_summary should call all policies with reset flag."""
    train_policy = _DummyPolicy("train")
    eval_policy = _DummyPolicy("eval")
    group = PolicyGroup([train_policy], [eval_policy], _make_policy_cfg())

    group.print_timing_summary(reset=True)

    assert train_policy.calls["print_timing_summary"] == [True]
    assert eval_policy.calls["print_timing_summary"] == [True]


def test_policy_group_init_eval_and_train_calls():
    """init_eval and init_train should call policy init methods."""
    train_policy = _DummyPolicy("train")
    eval_policy = _DummyPolicy("eval")
    group = PolicyGroup([train_policy], [eval_policy], _make_policy_cfg())

    group.init_eval(eval_config="cfg", env_meta="meta")
    group.init_train(train_config=TrainConfig(max_epochs=1), env_meta="meta")

    assert train_policy.calls["init_train"]
    assert eval_policy.calls["init_eval"]


def test_policy_group_init_comm_group_empty_is_noop():
    """init_comm_group should return early when no policies exist."""
    group = PolicyGroup([], [], _make_policy_cfg())
    group.init_comm_group(backend="gloo")


def test_policy_group_init_comm_group_remote_with_transfer(monkeypatch):
    """init_comm_group should wire up remote groups and barriers."""

    class _RemotePolicy:
        def __init__(
            self,
            name: str,
            node_id: str,
            gpu_id: int,
            rank: int,
            addr_port: tuple[str, int],
        ) -> None:
            self.name = name
            self._get_node_id = _RemoteMethod(lambda: node_id)
            self._get_gpu_ids = _RemoteMethod(lambda: [gpu_id])
            self._get_addr_and_port = _RemoteMethod(lambda: addr_port, wrap_ref=True)
            self.init_distributed_env = _RemoteMethod(lambda **kwargs: kwargs, wrap_ref=True)
            self.get_rank = _RemoteMethod(lambda: rank, wrap_ref=True)
            self.init_single_comm_group = _RemoteMethod(
                lambda ranks, mode, backend: (ranks, mode, backend), wrap_ref=True
            )
            self.dist_barrier = _RemoteMethod(lambda *args: "barrier", wrap_ref=True)

    train_policy = _RemotePolicy("train", "node-a", 0, 0, ("1.2.3.4", 1234))
    eval_policy = _RemotePolicy("eval", "node-b", 1, 1, ("9.9.9.9", 9999))

    group = PolicyGroup([train_policy], [eval_policy], _make_policy_cfg())
    group.weight_buffer_strategy = "Double"
    group.transfer_info = {
        "node-a": [
            {
                "sender": train_policy,
                "receiver": [eval_policy],
                "intra_node": False,
            }
        ]
    }

    def fake_ray_get(obj):
        if isinstance(obj, list):
            return [fake_ray_get(o) for o in obj]
        if isinstance(obj, _FakeObjectRef):
            return obj.value
        return obj

    monkeypatch.setattr("rlightning.policy.policy_group.ray.get", fake_ray_get)
    group.init_comm_group(backend="gloo")

    assert len(train_policy.init_distributed_env.calls) == 1
    assert len(eval_policy.init_distributed_env.calls) == 1

    _, train_kwargs = train_policy.init_distributed_env.calls[0]
    _, eval_kwargs = eval_policy.init_distributed_env.calls[0]

    assert train_kwargs["world_size"] == 2
    assert eval_kwargs["world_size"] == 2
    assert train_kwargs["master_addr"] == "1.2.3.4"
    assert train_kwargs["master_port"] == 1234
    assert train_kwargs["rank"] == 0
    assert eval_kwargs["rank"] == 1
    assert train_kwargs["local_rank"] == 0
    assert eval_kwargs["local_rank"] == 0
    assert train_kwargs["local_world_size"] == 1
    assert eval_kwargs["local_world_size"] == 1

    for policy in (train_policy, eval_policy):
        modes = [args[1] for args, _ in policy.init_single_comm_group.calls]
        assert ParallelMode.INTRA_NODE in modes
        assert ParallelMode.WEIGHT_TRANSFER in modes
        assert ParallelMode.TRAIN_DATA_PARALLEL in modes

    transfer_calls = [
        args for args, _ in train_policy.init_single_comm_group.calls if args[1] == ParallelMode.WEIGHT_TRANSFER
    ]
    assert transfer_calls[0][0] == [0, 1]

    assert len(train_policy.dist_barrier.calls) == 3
    assert len(eval_policy.dist_barrier.calls) == 3


def test_policy_group_push_pop_updates_lists():
    """push/pop should update role lists."""
    group = PolicyGroup([], [], _make_policy_cfg())
    train_policy = _DummyPolicy("train")
    eval_policy = _DummyPolicy("eval")

    group.push(train_policy, PolicyRole.TRAIN)
    group.push(eval_policy, PolicyRole.EVAL)

    assert group.train_list[-1] is train_policy
    assert group.eval_list[-1] is eval_policy

    popped = group.pop(PolicyRole.EVAL)
    assert popped is eval_policy
    assert group.eval_list == []


def test_policy_group_init_placement_info_local(monkeypatch):
    """init_placement_info should assign local node for local policies."""
    train_policy = _DummyPolicy("train")
    eval_policy = _DummyPolicy("eval")
    group = PolicyGroup([train_policy], [eval_policy], _make_policy_cfg(buffer_strategy="Double"))

    monkeypatch.setenv("RLIGHTNING_REMOTE_TRAIN", "0")
    monkeypatch.setenv("RLIGHTNING_REMOTE_EVAL", "0")

    group.init_placement_info()

    assert "local" in group.placement_info
    assert train_policy in group.placement_info["local"]["train_policies"]
    assert eval_policy in group.placement_info["local"]["eval_policies"]
    assert group.transfer_info["local"][0]["sender"] is train_policy
    assert group.transfer_info["local"][0]["receiver"] == [eval_policy]
    assert group.transfer_info["local"][0]["intra_node"] is True


def test_policy_group_init_placement_info_remote(monkeypatch):
    """init_placement_info should query node ids via ray for remote policies."""

    class _RemotePolicy:
        def __init__(self, name: str, node_id: str) -> None:
            self.name = name
            self._get_node_id = _RemoteMethod(lambda: node_id)

    train_policy = _RemotePolicy("train", "node-train")
    eval_policy = _RemotePolicy("eval", "node-eval")
    group = PolicyGroup([train_policy], [eval_policy], _make_policy_cfg(buffer_strategy="Double"))

    monkeypatch.setenv("RLIGHTNING_REMOTE_TRAIN", "1")
    monkeypatch.setenv("RLIGHTNING_REMOTE_EVAL", "1")

    def fake_ray_get(obj):
        if isinstance(obj, list):
            return [fake_ray_get(o) for o in obj]
        if isinstance(obj, _FakeObjectRef):
            return obj.value
        return obj

    monkeypatch.setattr("rlightning.policy.policy_group.ray.get", fake_ray_get)
    group.init_placement_info()

    assert "node-train" in group.placement_info
    assert "node-eval" in group.placement_info
    assert train_policy in group.placement_info["node-train"]["train_policies"]
    assert eval_policy in group.placement_info["node-eval"]["eval_policies"]
    assert group.transfer_info["node-train"][0]["receiver"] == [eval_policy]
    assert group.transfer_info["node-train"][0]["intra_node"] is False


def test_policy_group_init_placement_info_colocated():
    """init_placement_info should use colocated helper when configured."""
    train_policy = _DummyPolicy("train")
    eval_policy = _DummyPolicy("eval")
    group = PolicyGroup(
        [train_policy],
        [eval_policy],
        _make_policy_cfg(buffer_strategy="None"),
    )

    group.init_placement_info(is_colocated=True)

    assert group.transfer_info == {"all": [{"sender": train_policy, "receiver": eval_policy}]}
    assert group.placement_info == {}


def test_policy_group_init_weight_buffer_double():
    """init_weight_buffer should call init on eval policies for Double strategy."""
    eval_policy = _DummyPolicy("eval")
    group = PolicyGroup([], [eval_policy], _make_policy_cfg(buffer_strategy="Double"))

    group.init_weight_buffer()

    assert eval_policy.calls["init_weight_buffer"] == [None]


def test_policy_group_init_weight_buffer_none_noop():
    """init_weight_buffer should skip when strategy is None."""
    eval_policy = _DummyPolicy("eval")
    group = PolicyGroup([], [eval_policy], _make_policy_cfg(buffer_strategy="None"))

    group.init_weight_buffer()

    assert eval_policy.calls["init_weight_buffer"] == []


def test_policy_group_init_weight_buffer_invalid_strategy_raises():
    """init_weight_buffer should reject unsupported strategies."""
    eval_policy = _DummyPolicy("eval")
    group = PolicyGroup([], [eval_policy], _make_policy_cfg(buffer_strategy="Double"))
    group.config.weight_buffer.buffer_strategy = "BadStrategy"

    with pytest.raises(ValueError):
        group.init_weight_buffer()


def test_policy_group_init_weight_buffer_shared(monkeypatch):
    """Shared strategy should create shared buffer and reuse across eval policies."""
    eval_policy = _DummyPolicy("eval")
    group = PolicyGroup(
        [],
        [eval_policy],
        _make_policy_cfg(buffer_strategy="Shared", buffer_type="CPUWeightBuffer"),
    )
    group.placement_info = {"local": {"train_policies": [], "eval_policies": [eval_policy]}}

    monkeypatch.setattr(
        "rlightning.policy.policy_group.build_weight_buffer",
        lambda *args, **kwargs: "shared-buffer",
    )

    group.init_weight_buffer()

    assert eval_policy.calls["init_weight_buffer"] == ["shared-buffer"]
    assert group.shared_weight_buffer_map["local"]["shared_weight_buffer"] == "shared-buffer"


def test_build_transfer_info_for_disaggregated_various_strategies():
    """build_transfer_info_for_disaggregated should route train/eval nodes by strategy."""
    train = _DummyPolicy("train")
    eval_a = _DummyPolicy("eval-a")
    eval_b = _DummyPolicy("eval-b")

    placement = {
        "node-train": {"train_policies": [train], "eval_policies": []},
        "node-eval": {"train_policies": [], "eval_policies": [eval_a, eval_b]},
    }

    transfer_info = build_transfer_info_for_disaggregated(placement, "Double")
    assert transfer_info["node-train"][0]["receiver"] == [eval_a, eval_b]
    assert transfer_info["node-train"][0]["intra_node"] is False

    transfer_info = build_transfer_info_for_disaggregated(placement, "Shared")
    assert transfer_info["node-train"][0]["receiver"] == [eval_a]

    transfer_info = build_transfer_info_for_disaggregated(placement, "None")
    assert transfer_info["node-train"][0]["receiver"] == [eval_a, eval_b]

    with pytest.raises(ValueError):
        build_transfer_info_for_disaggregated(
            {"node-eval": {"train_policies": [], "eval_policies": [eval_a]}},
            "Double",
        )


def test_build_transfer_info_for_colocated():
    """build_transfer_info_for_colocated should map train/eval policy pairs by index."""
    train_policies = [_DummyPolicy("train-0"), _DummyPolicy("train-1")]
    eval_policies = [_DummyPolicy("eval-0"), _DummyPolicy("eval-1")]

    transfer_info = build_transfer_info_for_colocated(train_policies, eval_policies)

    assert transfer_info["all"][0]["sender"] is train_policies[0]
    assert transfer_info["all"][0]["receiver"] is eval_policies[0]
    assert transfer_info["all"][1]["sender"] is train_policies[1]
    assert transfer_info["all"][1]["receiver"] is eval_policies[1]


def test_policy_group_send_weights_double():
    """send_weights should call train send_weights without shared buffer."""
    train_policy = _DummyPolicy("train")
    eval_policy = _DummyPolicy("eval")
    group = PolicyGroup([train_policy], [eval_policy], _make_policy_cfg())
    group.weight_buffer_strategy = "Double"
    group.transfer_info = {
        "local": [
            {
                "sender": train_policy,
                "receiver": [eval_policy],
                "intra_node": False,
            }
        ]
    }

    group.send_weights()

    assert train_policy.calls["send_weights"] == [([eval_policy], None)]
    assert train_policy.calls["save_checkpoint"] == []


def test_policy_group_send_weights_remote_shared(monkeypatch):
    """send_weights should use remote calls when RLIGHTNING_REMOTE_TRAIN is set."""

    class _RemoteTrainPolicy:
        def __init__(self) -> None:
            self.send_weights = _RemoteMethod(
                lambda receivers, shared_weight_buffer=None: "sent",
                wrap_ref=True,
            )
            self.save_weights = _RemoteMethod(lambda save_dir, epoch: "saved", wrap_ref=True)

    train_policy = _RemoteTrainPolicy()
    eval_policy = _DummyPolicy("eval")

    group = PolicyGroup([train_policy], [eval_policy], _make_policy_cfg())
    group.weight_buffer_strategy = "Shared"
    group.transfer_info = {
        "local": [
            {
                "sender": train_policy,
                "receiver": [eval_policy],
                "intra_node": True,
            }
        ]
    }
    group.shared_weight_buffer_map = {"local": {"shared_weight_buffer": "buf"}}

    monkeypatch.setenv("RLIGHTNING_REMOTE_TRAIN", "1")

    def fake_ray_get(obj):
        if isinstance(obj, list):
            return [fake_ray_get(o) for o in obj]
        if isinstance(obj, _FakeObjectRef):
            return obj.value
        return obj

    monkeypatch.setattr("rlightning.policy.policy_group.ray.get", fake_ray_get)
    group.send_weights()

    assert train_policy.send_weights.calls[0][0] == ([eval_policy], "buf")


def test_policy_group_send_weights_shared():
    """send_weights should use shared buffer when configured for local mode."""
    train_policy = _DummyPolicy("train")
    eval_policy = _DummyPolicy("eval")

    group = PolicyGroup([train_policy], [eval_policy], _make_policy_cfg())
    group.weight_buffer_strategy = "Shared"
    group.transfer_info = {
        "local": [
            {
                "sender": train_policy,
                "receiver": [eval_policy],
                "intra_node": True,
            }
        ]
    }
    group.shared_weight_buffer_map = {"local": {"shared_weight_buffer": "buf"}}

    group.send_weights()

    assert train_policy.calls["send_weights"][0] == ([eval_policy], "buf")


def test_policy_group_save_checkpoint_local(tmp_path):
    """save_checkpoint should call local save_checkpoint directly."""
    train_policy = _DummyPolicy("train")
    group = PolicyGroup([train_policy], [], _make_policy_cfg())

    group.save_checkpoint(str(tmp_path))

    assert train_policy.calls["save_checkpoint"] == [str(tmp_path)]


def test_policy_group_save_checkpoint_remote(monkeypatch, tmp_path):
    """save_checkpoint should call remote save_checkpoint when remote train is enabled."""

    class _RemoteTrainPolicy:
        def __init__(self) -> None:
            self.save_checkpoint = _RemoteMethod(lambda path: "saved", wrap_ref=True)

    train_policy = _RemoteTrainPolicy()
    group = PolicyGroup([train_policy], [], _make_policy_cfg())
    monkeypatch.setenv("RLIGHTNING_REMOTE_TRAIN", "1")

    def fake_submit(method, *args, _block: bool = False, **kwargs):
        # Simulate TaskSubmitter behavior for remote actors by always calling .remote
        return method.remote(*args, **kwargs)

    def fake_ray_get(obj):
        if isinstance(obj, _FakeObjectRef):
            return obj.value
        return obj

    # Route PolicyGroup.save_checkpoint through our fake submitter and fake ray.get
    group._eval_task_submitter.submit = fake_submit
    group._train_task_submitter.submit = fake_submit
    monkeypatch.setattr("rlightning.policy.policy_group.ray.get", fake_ray_get)
    group.save_checkpoint(str(tmp_path))

    # _RemoteMethod records calls as (args, kwargs); first arg of first call is the path
    assert train_policy.save_checkpoint.calls[0][0][0] == str(tmp_path)


def test_policy_group_notify_update_weights():
    """notify_update_weights should signal all eval policies."""
    eval_policies = [_DummyPolicy("e1"), _DummyPolicy("e2")]
    group = PolicyGroup([], eval_policies, _make_policy_cfg())

    group.notify_update_weights()

    assert eval_policies[0].calls["notify_update_weights"] == 1
    assert eval_policies[1].calls["notify_update_weights"] == 1
