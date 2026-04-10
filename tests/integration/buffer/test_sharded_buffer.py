import os

import pytest
import ray
from tensordict import TensorDict

from rlightning.types import BatchedData, PolicyResponse
from rlightning.utils.builders import build_data_buffer, build_env_group
from rlightning.utils.config import BufferConfig, ClusterConfig
from rlightning.utils.placement import GlobalResourceManager
from rlightning.utils.placement.placement_strategies import (
    PLACEMENT_STRATEGIES,
    DefaultPlacementStrategy,
)
from rlightning.utils.placement.scheduling import ComponentScheduling, Scheduling
from rlightning.utils.ray import resolve_object
from tests.test_utils import _make_env_config, require_env_backend, setup_ray_cluster, teardown_ray_cluster

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _auto_ray_cluster():
    setup_ray_cluster()
    yield
    teardown_ray_cluster()


class _ShardedPlacementStrategy(DefaultPlacementStrategy):
    """
    Placement strategy for testing sharded buffer on single node.
    It allows placing multiple shards on the same node (round-robin).
    """

    def create_placement_groups(self):
        num_buffer_storages = self.scheduling.buffer_worker.worker_num
        if num_buffer_storages > 1:
            node_ids = list(self._node_info["node_id_to_resources"].keys())

            for storage_index in range(num_buffer_storages):
                node_id = node_ids[storage_index % len(node_ids)]
                self.buffer_strategies.append(
                    ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=False,
                    )
                )
        return {}


# TODO: now only test sharded storage of ReplayBuffer type
def _make_sharded_buffer_config(capacity: int = 100, num_shards: int = 2) -> BufferConfig:
    os.environ["RLIGHTNING_REMOTE_STORAGE"] = "1"
    return BufferConfig.from_dict(
        {
            "type": "ReplayBuffer",
            "capacity": capacity,
            "storage": {
                "type": "sharded",
                "device": "cpu",
            },
            "sampler": {
                "type": "uniform",
            },
        }
    )


def _init_global_resource_manager(num_shards=2, num_train_workers=2):
    if "test_sharded" not in PLACEMENT_STRATEGIES:
        PLACEMENT_STRATEGIES["test_sharded"] = _ShardedPlacementStrategy

    global_resource_manager = GlobalResourceManager.get_instance()
    placement_cfg = ClusterConfig()
    placement_cfg.placement.strategy = "test_sharded"

    scheduling = ComponentScheduling(
        env_worker=[Scheduling(worker_num=1, num_cpus=0, num_gpus=0)],
        train_worker=Scheduling(worker_num=num_train_workers, num_cpus=0, num_gpus=0),
        eval_worker=Scheduling(worker_num=1, num_cpus=0, num_gpus=0),
        buffer_worker=Scheduling(worker_num=num_shards, num_cpus=0, num_gpus=0),
    )

    global_resource_manager.initialize(placement_cfg, scheduling)
    return global_resource_manager


def _reset_global_resource_manager():
    global_resource_manager = GlobalResourceManager.get_instance()
    global_resource_manager.reset()


def _mock_batched_policy_ret(action_spaces, env_ids):
    action_list = [resolve_object(action_space).sample() for action_space in action_spaces]
    policy_resp_list = [
        PolicyResponse(env_id=env_id, action=action, log_prob=0.0, entropy=0.0)
        for action, env_id in zip(action_list, env_ids)
    ]
    batched_policy_resp = BatchedData(env_ids, policy_resp_list)
    return batched_policy_resp


def test_sharded_init():
    num_shards = 2
    num_train_workers = 2
    _init_global_resource_manager(num_shards=num_shards, num_train_workers=num_train_workers)
    buffer_cfg = _make_sharded_buffer_config(num_shards=num_shards)
    env_ids = [f"env_{i}" for i in range(4)]

    buffer = build_data_buffer(buffer_cls=buffer_cfg.type, buffer_cfg=buffer_cfg)
    buffer.init(env_meta_list=None, env_ids=env_ids)

    assert len(buffer.storages) == num_shards
    assert buffer.size() == 0

    _reset_global_resource_manager()


@pytest.mark.parametrize("num_env_workers", [2, 4])
@pytest.mark.parametrize("num_train_workers", [2, 4])
@pytest.mark.parametrize("num_shards", [2])
@pytest.mark.parametrize("steps", [10])
def test_sharded_buffer(num_env_workers, num_train_workers, num_shards, steps):
    require_env_backend("mujoco")
    _init_global_resource_manager(num_shards=num_shards, num_train_workers=num_train_workers)

    buffer_cfg = _make_sharded_buffer_config(capacity=100, num_shards=num_shards)

    env_cfg = _make_env_config(env_backend="mujoco", task_name="Ant-v5", num_workers=num_env_workers)
    env_group = build_env_group([env_cfg])
    batched_env_ret, _ = env_group.reset()
    env_ids = batched_env_ret.ids()

    buffer = build_data_buffer(buffer_cls=buffer_cfg.type, buffer_cfg=buffer_cfg)
    buffer.init(env_meta_list=None, env_ids=env_ids)

    # Add transitions
    for _ in range(steps):
        batched_policy_resp = _mock_batched_policy_ret(env_group.get_action_spaces(), batched_env_ret.ids())
        buffer.add_batched_transition(batched_env_ret, batched_policy_resp)
        batched_env_ret, _ = env_group.step(batched_policy_resp)

    # Truncate episodes to flush to storage
    truncations = [True] * len(batched_env_ret)
    batched_policy_resp = _mock_batched_policy_ret(env_group.get_action_spaces(), batched_env_ret.ids())
    buffer.add_batched_transition(batched_env_ret, batched_policy_resp, truncations=truncations)

    # Check total size
    expected_size = num_env_workers * steps
    assert buffer.size() == expected_size

    # Test sampling
    batch_size = num_env_workers * steps
    sample_data = buffer.sample(batch_size=batch_size)

    assert len(sample_data) == num_train_workers

    total_samples = 0
    for data in sample_data:
        data = resolve_object(data)
        data = TensorDict.from_dict(data, auto_batch_size=True)
        total_samples += len(data)
    assert total_samples == batch_size

    _reset_global_resource_manager()


@pytest.mark.parametrize("num_env_workers", [2, 4])
@pytest.mark.parametrize("num_train_workers", [2, 4])
@pytest.mark.parametrize("num_shards", [2])
@pytest.mark.parametrize("steps", [10])
def test_add_data_async(num_env_workers, num_train_workers, num_shards, steps):
    require_env_backend("mujoco")
    _init_global_resource_manager(num_shards=num_shards, num_train_workers=num_train_workers)

    buffer_cfg = _make_sharded_buffer_config(capacity=100, num_shards=num_shards)

    env_cfg = _make_env_config(env_backend="mujoco", task_name="Ant-v5", num_workers=num_env_workers)
    env_group = build_env_group([env_cfg])
    batched_env_ret, _ = env_group.reset()
    env_ids = batched_env_ret.ids()

    buffer = build_data_buffer(buffer_cls=buffer_cfg.type, buffer_cfg=buffer_cfg)
    buffer.init(env_meta_list=None, env_ids=env_ids)

    # Add transitions
    for _ in range(steps):
        batched_policy_resp = _mock_batched_policy_ret(env_group.get_action_spaces(), batched_env_ret.ids())
        buffer.add_batched_data_async(batched_env_ret)
        buffer.add_batched_data_async(batched_policy_resp)

        batched_env_ret, _ = env_group.step(batched_policy_resp)

    assert buffer.size() == 0

    # Explicitly truncate
    batched_policy_resp = _mock_batched_policy_ret(env_group.get_action_spaces(), batched_env_ret.ids())
    buffer.add_batched_data_async(batched_env_ret)
    buffer.add_batched_data_async(batched_policy_resp)
    buffer.truncate_episodes(env_ids)

    # Check total size
    expected_size = num_env_workers * steps
    assert buffer.size() == expected_size

    _reset_global_resource_manager()
