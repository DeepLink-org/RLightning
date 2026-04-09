from __future__ import annotations

from pathlib import Path

import ray
import pytest
import yaml

from rlightning.utils.config import ClusterConfig
from rlightning.utils.placement import ComponentScheduling, GlobalResourceManager, ResourcePoolPlanner, Scheduling
import rlightning.utils.placement.placement_manager as placement_manager_module
import rlightning.utils.placement.placement_strategies as placement_strategies_module
import rlightning.utils.placement.resource_pool as resource_pool_module


def create_mock_cluster_resources(
    num_nodes: int = 2,
    gpus_per_node: int = 8,
    cpus_per_node: int = 64,
):
    node_id_to_resources = {}
    for i in range(num_nodes):
        node_id = f"node_{i}"
        node_id_to_resources[node_id] = {
            "node_id": node_id,
            "ip": f"192.168.1.{i + 1}",
            "CPU": cpus_per_node,
            "GPU": gpus_per_node,
        }
    return {
        "node_id_to_resources": node_id_to_resources,
        "total_cpus": num_nodes * cpus_per_node,
        "total_gpus": num_nodes * gpus_per_node,
    }


def create_mock_scheduling(
    train_workers: int = 0,
    train_gpus: float = 1.0,
    eval_workers: int = 0,
    eval_gpus: float = 1.0,
    buffer_workers: int | str = 0,
    buffer_gpus: float = 0.0,
    env_workers: int = 0,
    env_gpus: float = 1.0,
) -> ComponentScheduling:
    return ComponentScheduling(
        train_worker=Scheduling(worker_num=train_workers, num_cpus=1, num_gpus=train_gpus),
        eval_worker=Scheduling(worker_num=eval_workers, num_cpus=1, num_gpus=eval_gpus),
        buffer_worker=Scheduling(worker_num=buffer_workers, num_cpus=1, num_gpus=buffer_gpus),
        env_worker=[Scheduling(worker_num=env_workers, num_cpus=1, num_gpus=env_gpus)],
    )


def create_mock_cluster_config(
    strategy: str = "default",
    env_strategy: str = "default",
    mode: str = "auto",
    resource_pool: list[dict] | None = None,
    train_worker_num: int = 4,
    eval_worker_num: int = 2,
    buffer_worker_num: int | str = 1,
) -> ClusterConfig:
    cluster_dict = {
        "placement": {
            "strategy": strategy,
            "env_strategy": env_strategy,
            "mode": mode,
        },
        "train_worker_num": train_worker_num,
        "eval_worker_num": eval_worker_num,
        "buffer_worker_num": buffer_worker_num,
    }
    if resource_pool is not None:
        cluster_dict["resource_pool"] = resource_pool
    return ClusterConfig.from_dict(cluster_dict)


class _FakePlacementGroup:
    def __init__(self, bundles, name, strategy, node_id):
        self.bundles = bundles
        self.name = name
        self.strategy = strategy
        self.node_id = node_id
        self.id = f"{name}-id"

    def ready(self):
        return True


def patch_cluster_sources(monkeypatch, cluster_info):
    monkeypatch.setattr(placement_manager_module, "get_cluster_resources", lambda: cluster_info)
    monkeypatch.setattr(resource_pool_module, "get_cluster_resources", lambda: cluster_info)
    monkeypatch.setattr(placement_strategies_module, "get_cluster_resources", lambda: cluster_info)


def patch_ray_runtime(monkeypatch):
    created = {}

    def _placement_group(bundles, name, strategy, _soft_target_node_id=None):
        pg = _FakePlacementGroup(bundles, name, strategy, _soft_target_node_id)
        created[name] = pg
        return pg

    monkeypatch.setattr(ray.util, "placement_group", _placement_group)
    monkeypatch.setattr(ray, "get", lambda _ready: None)
    return created


@pytest.fixture
def mock_cluster_2_nodes_8_gpus():
    return create_mock_cluster_resources(num_nodes=2, gpus_per_node=8)


@pytest.fixture
def mock_cluster_1_node_8_gpus():
    return create_mock_cluster_resources(num_nodes=1, gpus_per_node=8)


@pytest.fixture
def basic_scheduling():
    return create_mock_scheduling(
        train_workers=4,
        train_gpus=1.0,
        eval_workers=2,
        eval_gpus=1.0,
        buffer_workers=1,
        env_workers=4,
        env_gpus=0.5,
    )


@pytest.fixture
def medium_scheduling():
    return create_mock_scheduling(
        train_workers=8,
        train_gpus=1.0,
        eval_workers=4,
        eval_gpus=1.0,
        buffer_workers=1,
        env_workers=8,
        env_gpus=0.5,
    )


@pytest.fixture
def large_scheduling():
    return create_mock_scheduling(
        train_workers=8,
        train_gpus=1.0,
        eval_workers=8,
        eval_gpus=1.0,
        buffer_workers=1,
        env_workers=16,
        env_gpus=0.5,
    )


def test_resource_pool_planner_discovers_cluster_resources(mock_cluster_2_nodes_8_gpus, basic_scheduling):
    planner = ResourcePoolPlanner(scheduling=basic_scheduling, cluster_info=mock_cluster_2_nodes_8_gpus)

    nodes = planner.discover_cluster_resources()

    assert set(nodes) == {"node_0", "node_1"}
    assert nodes["node_0"].total_gpus == 8
    assert nodes["node_1"].total_cpus == 64


def test_resource_pool_planner_validates_disaggregate_scheduling(
    mock_cluster_2_nodes_8_gpus, basic_scheduling
):
    planner = ResourcePoolPlanner(scheduling=basic_scheduling, cluster_info=mock_cluster_2_nodes_8_gpus)

    is_valid, error_msg = planner.validate_scheduling(strategy="disaggregate")

    assert is_valid is True
    assert error_msg == ""
    assert planner.resource_summary == {
        "train_pool_required_gpus": 4.0,
        "rollout_pool_required_gpus": 4.0,
        "env_required_gpus": 2.0,
        "eval_required_gpus": 2.0,
    }


def test_resource_pool_planner_rejects_insufficient_gpus(mock_cluster_1_node_8_gpus, large_scheduling):
    planner = ResourcePoolPlanner(scheduling=large_scheduling, cluster_info=mock_cluster_1_node_8_gpus)

    is_valid, error_msg = planner.validate_scheduling(strategy="disaggregate")

    assert is_valid is False
    assert "Insufficient GPUs" in error_msg


def test_resource_pool_planner_plans_disaggregate_pools(mock_cluster_2_nodes_8_gpus, medium_scheduling):
    planner = ResourcePoolPlanner(scheduling=medium_scheduling, cluster_info=mock_cluster_2_nodes_8_gpus)

    pools = planner.plan_resource_pools(strategy="disaggregate")

    assert set(pools) == {"train_pool", "rollout_pool"}
    assert set(pools["train_pool"].component_types) == {"buffer", "train"}
    assert pools["train_pool"].get_component_indices("train") == "0-7"
    assert set(pools["rollout_pool"].component_types) == {"env", "eval"}
    assert pools["rollout_pool"].get_component_indices("eval") == "0-3"
    assert pools["rollout_pool"].get_component_indices("env") == "4-7"
    assert planner.get_pool_for_component("train").name == "train_pool"
    assert planner.get_pool_for_component("eval").name == "rollout_pool"


def test_resource_pool_planner_loads_manual_resource_pools(mock_cluster_2_nodes_8_gpus, basic_scheduling):
    planner = ResourcePoolPlanner(scheduling=basic_scheduling, cluster_info=mock_cluster_2_nodes_8_gpus)
    planner.discover_cluster_resources()

    pools = planner.load_manual_resource_pools(
        [
            {
                "name": "train_pool",
                "num_node": 1,
                "num_gpus": 8,
                "train": "0-7",
            },
            {
                "name": "rollout_pool",
                "num_node": 1,
                "num_gpus": 8,
                "eval": "0-3",
                "env": "4-7",
            },
        ]
    )

    assert set(pools) == {"train_pool", "rollout_pool"}
    assert set(pools["train_pool"].component_types) == {"buffer", "train"}
    assert set(pools["rollout_pool"].component_types) == {"env", "eval"}


def test_resource_pool_planner_to_yaml_and_summary(mock_cluster_2_nodes_8_gpus, medium_scheduling):
    planner = ResourcePoolPlanner(scheduling=medium_scheduling, cluster_info=mock_cluster_2_nodes_8_gpus)
    planner.plan_resource_pools(strategy="disaggregate")

    yaml_config = planner.to_yaml_config()
    summary = planner.summary()

    assert len(yaml_config) == 2
    assert set(yaml_config[0].keys()) == {"name", "num_node", "num_gpus", "train"}
    assert set(yaml_config[1].keys()) == {"name", "num_node", "num_gpus", "eval", "env"}
    assert set(summary) == {"cluster", "pools", "yaml_config"}
    assert summary["yaml_config"] == yaml_config


def test_global_resource_manager_is_singleton():
    manager1 = GlobalResourceManager.get_instance()
    manager2 = GlobalResourceManager.get_instance()

    assert manager1 is manager2
    assert manager1.is_initialized is False


def test_global_resource_manager_requires_initialize_for_runtime_methods():
    manager = GlobalResourceManager.get_instance()

    with pytest.raises(RuntimeError, match="not initialized"):
        manager.get_scheduling_strategy("train", 0)

    with pytest.raises(RuntimeError, match="not initialized"):
        manager.get_storage_to_train_workers()

    with pytest.raises(RuntimeError, match="not initialized"):
        manager.save_yaml_config("/tmp")

    assert manager.get_placement_config() is None
    assert manager.get_scheduling() is None
    assert manager.get_pool_for_component("train") is None


def test_global_resource_manager_initializes_disaggregate_strategy(
    monkeypatch, mock_cluster_2_nodes_8_gpus, medium_scheduling
):
    patch_cluster_sources(monkeypatch, mock_cluster_2_nodes_8_gpus)
    created = patch_ray_runtime(monkeypatch)

    manager = GlobalResourceManager.get_instance()
    manager.initialize(create_mock_cluster_config(strategy="disaggregate"), medium_scheduling)

    assert manager.is_initialized is True
    assert manager.get_placement_strategy() == "disaggregate"
    assert set(manager.get_resource_pools()) == {"train_pool", "rollout_pool"}
    assert manager.get_pool_for_component("train").name == "train_pool"
    assert manager.get_pool_for_component("eval").name == "rollout_pool"
    assert manager.get_storage_to_train_workers() == {0: list(range(8))}
    assert manager.get_scheduling_strategy("train", 0) != "DEFAULT"
    assert set(created) == {"train_pool_node_0", "rollout_pool_node_1"}


def test_global_resource_manager_initializes_manual_mode(monkeypatch, mock_cluster_2_nodes_8_gpus, basic_scheduling):
    patch_cluster_sources(monkeypatch, mock_cluster_2_nodes_8_gpus)
    patch_ray_runtime(monkeypatch)

    resource_pool = [
        {
            "name": "train_pool",
            "num_node": 1,
            "num_gpus": 8,
            "train": "0-7",
        },
        {
            "name": "rollout_pool",
            "num_node": 1,
            "num_gpus": 8,
            "eval": "0-3",
            "env": "4-7",
        },
    ]

    manager = GlobalResourceManager.get_instance()
    manager.initialize(
        create_mock_cluster_config(mode="manual", resource_pool=resource_pool),
        basic_scheduling,
    )

    assert manager.is_initialized is True
    assert manager.get_placement_strategy() == "resource_pool"
    assert set(manager.get_resource_pools()) == {"train_pool", "rollout_pool"}
    assert manager.get_pool_for_component("train").name == "train_pool"
    assert manager.get_pool_for_component("eval").name == "rollout_pool"


def test_global_resource_manager_saves_yaml_config(monkeypatch, tmp_path, mock_cluster_2_nodes_8_gpus, medium_scheduling):
    patch_cluster_sources(monkeypatch, mock_cluster_2_nodes_8_gpus)
    patch_ray_runtime(monkeypatch)

    manager = GlobalResourceManager.get_instance()
    manager.initialize(create_mock_cluster_config(strategy="disaggregate"), medium_scheduling)

    saved_path = Path(manager.save_yaml_config(str(tmp_path)))
    saved_config = yaml.safe_load(saved_path.read_text())

    assert saved_path.exists()
    assert saved_path.name == "resource_pool_auto.yaml"
    assert len(saved_config) == 2
    assert saved_config[0]["name"] == "train_pool"
    assert saved_config[1]["name"] == "rollout_pool"
