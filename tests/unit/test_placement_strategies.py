from unittest.mock import MagicMock, patch

import pytest

from rlightning.utils.placement import ResourcePoolPlanner
from rlightning.utils.placement.placement_strategies import (
    ColocatedPlacementStrategy,
    DefaultPlacementStrategy,
    DisaggregatePlacementStrategy,
    _pack_workers_on_gpu_units,
)
from rlightning.utils.placement.scheduling import ComponentScheduling, Scheduling


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


@pytest.fixture
def mock_cluster_2_nodes_8_gpus():
    return create_mock_cluster_resources(num_nodes=2, gpus_per_node=8)


@pytest.fixture
def mock_cluster_1_node_8_gpus():
    return create_mock_cluster_resources(num_nodes=1, gpus_per_node=8)


@pytest.fixture
def mock_cluster_3_nodes_8_gpus():
    return create_mock_cluster_resources(num_nodes=3, gpus_per_node=8)


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
def large_train_scheduling():
    return create_mock_scheduling(
        train_workers=16,
        train_gpus=1.0,
        eval_workers=4,
        eval_gpus=1.0,
        buffer_workers="auto",
        env_workers=8,
        env_gpus=0.5,
    )


def test_pack_workers_basic():
    worker_locations = []
    unit_cpu = [0, 0, 0, 0]
    distribution = {}

    workers_placed = _pack_workers_on_gpu_units(
        allocations=[(0, 3)],
        node_id="node-0",
        component_type="train",
        node_component_distribution=distribution,
        pg_key="test_pg",
        capacity_gpus=4,
        unit_cpu=unit_cpu,
        worker_locations=worker_locations,
        workers_placed=0,
        workers_total=4,
        gpu_req_list=[1.0, 1.0, 1.0, 1.0],
        cpu_req=1,
    )

    assert workers_placed == 4
    assert len(worker_locations) == 4
    assert all(loc[0] == "test_pg" for loc in worker_locations)
    assert distribution["node-0"]["train"] == {"count": 4, "ids": [0, 1, 2, 3]}


def test_pack_workers_fractional_gpu():
    worker_locations = []
    unit_cpu = [0, 0]
    distribution = {}

    workers_placed = _pack_workers_on_gpu_units(
        allocations=[(0, 1)],
        node_id="node-0",
        component_type="eval",
        node_component_distribution=distribution,
        pg_key="test_pg",
        capacity_gpus=2,
        unit_cpu=unit_cpu,
        worker_locations=worker_locations,
        workers_placed=0,
        workers_total=4,
        gpu_req_list=[0.5, 0.5, 0.5, 0.5],
        cpu_req=1,
    )

    assert workers_placed == 4
    assert distribution["node-0"]["eval"]["count"] == 4


def test_pack_workers_invalid_unit_index():
    worker_locations = []
    unit_cpu = [0, 0]
    distribution = {}

    with pytest.raises(RuntimeError, match="Invalid GPU unit index"):
        _pack_workers_on_gpu_units(
            allocations=[(0, 5)],
            node_id="node-0",
            component_type="train",
            node_component_distribution=distribution,
            pg_key="test_pg",
            capacity_gpus=2,
            unit_cpu=unit_cpu,
            worker_locations=worker_locations,
            workers_placed=0,
            workers_total=2,
            gpu_req_list=[1.0, 1.0],
            cpu_req=1,
        )


@patch("ray.util.placement_group")
@patch("ray.get")
@patch("rlightning.utils.placement.resource_pool.get_cluster_resources")
def test_disaggregate_create_placement_groups(
    mock_pool_cluster,
    mock_ray_get,
    mock_placement_group,
    basic_scheduling,
    mock_cluster_2_nodes_8_gpus,
):
    mock_pool_cluster.return_value = mock_cluster_2_nodes_8_gpus
    mock_pg = MagicMock()
    mock_pg.ready.return_value = True
    mock_placement_group.return_value = mock_pg
    mock_ray_get.return_value = None

    strategy = DisaggregatePlacementStrategy(basic_scheduling)
    planner = ResourcePoolPlanner(scheduling=basic_scheduling)
    pools = planner.plan_resource_pools(strategy="disaggregate")

    train_node_count = planner.get_component_node_count("train")
    strategy.scheduling.adjust_buffer_worker_num(train_node_count)
    pgs = strategy.create_placement_groups(resource_pools=pools)

    assert "train_pool_node_0" in pgs
    assert strategy._storage_to_train_workers == {0: [0, 1, 2, 3]}
    assert [loc[1] for loc in strategy._worker_locations["train"]] == [0, 1, 2, 3]
    assert [loc[1] for loc in strategy._worker_locations["buffer"]] == [8]
    assert [loc[1] for loc in strategy._worker_locations["eval"]] == [4, 5]
    assert [loc[1] for loc in strategy._worker_locations["env"]] == [6, 6, 7, 7]


@patch("ray.util.placement_group")
@patch("ray.get")
@patch("rlightning.utils.placement.resource_pool.get_cluster_resources")
def test_colocated_create_placement_groups(
    mock_pool_cluster,
    mock_ray_get,
    mock_placement_group,
    basic_scheduling,
    mock_cluster_1_node_8_gpus,
):
    mock_pool_cluster.return_value = mock_cluster_1_node_8_gpus
    mock_pg = MagicMock()
    mock_pg.ready.return_value = True
    mock_placement_group.return_value = mock_pg
    mock_ray_get.return_value = None

    strategy = ColocatedPlacementStrategy(basic_scheduling)
    planner = ResourcePoolPlanner(scheduling=basic_scheduling)
    pools = planner.plan_resource_pools(strategy="colocate")

    train_node_count = planner.get_component_node_count("train")
    strategy.scheduling.adjust_buffer_worker_num(train_node_count)
    pgs = strategy.create_placement_groups(resource_pools=pools)

    assert strategy._storage_to_train_workers == {0: [0, 1, 2, 3]}
    assert "global_pool_node_0" in pgs
    assert [loc[1] for loc in strategy._worker_locations["train"]] == [0, 1, 2, 3]
    assert [loc[1] for loc in strategy._worker_locations["buffer"]] == [4]
    assert [loc[1] for loc in strategy._worker_locations["eval"]] == [0, 1]
    assert [loc[1] for loc in strategy._worker_locations["env"]] == [2, 2, 3, 3]


@patch("rlightning.utils.placement.placement_strategies.get_cluster_resources")
def test_default_strategy_create_placement_groups_single_buffer(
    mock_get_cluster, basic_scheduling, mock_cluster_2_nodes_8_gpus
):
    mock_get_cluster.return_value = mock_cluster_2_nodes_8_gpus

    strategy = DefaultPlacementStrategy(basic_scheduling)
    result = strategy.create_placement_groups()

    assert len(result) == 0


@patch("rlightning.utils.placement.placement_strategies.get_cluster_resources")
def test_default_strategy_create_placement_groups_multiple_buffers(mock_get_cluster, mock_cluster_2_nodes_8_gpus):
    mock_get_cluster.return_value = mock_cluster_2_nodes_8_gpus
    scheduling = create_mock_scheduling(buffer_workers=2)

    strategy = DefaultPlacementStrategy(scheduling)
    strategy.create_placement_groups()

    assert len(strategy.buffer_strategies) == 2


@patch("rlightning.utils.placement.placement_strategies.get_cluster_resources")
def test_default_strategy_returns_default_scheduling(mock_get_cluster, basic_scheduling, mock_cluster_2_nodes_8_gpus):
    mock_get_cluster.return_value = mock_cluster_2_nodes_8_gpus

    strategy = DefaultPlacementStrategy(basic_scheduling)
    strategy.create_placement_groups()

    assert strategy.get_scheduling_strategy("train", 0) == "DEFAULT"


@patch("rlightning.utils.placement.placement_strategies.get_cluster_resources")
def test_default_strategy_keeps_empty_storage_mapping(mock_get_cluster, mock_cluster_2_nodes_8_gpus):
    mock_get_cluster.return_value = mock_cluster_2_nodes_8_gpus
    scheduling = create_mock_scheduling(train_workers=4, buffer_workers=2)

    strategy = DefaultPlacementStrategy(scheduling)
    strategy.create_placement_groups()

    assert strategy.get_storage_to_train_workers() == {}


@patch("ray.util.placement_group")
@patch("ray.get")
@patch("rlightning.utils.placement.resource_pool.get_cluster_resources")
def test_disaggregate_train_workers_multi_node(
    mock_pool_cluster,
    mock_ray_get,
    mock_placement_group,
    mock_cluster_3_nodes_8_gpus,
    large_train_scheduling,
):
    mock_pool_cluster.return_value = mock_cluster_3_nodes_8_gpus
    mock_pg = MagicMock()
    mock_pg.ready.return_value = True
    mock_placement_group.return_value = mock_pg
    mock_ray_get.return_value = None

    scheduling = large_train_scheduling
    strategy = DisaggregatePlacementStrategy(scheduling)
    planner = ResourcePoolPlanner(scheduling=scheduling)
    pools = planner.plan_resource_pools(strategy="disaggregate")

    train_node_count = planner.get_component_node_count("train")
    scheduling.adjust_buffer_worker_num(train_node_count)
    strategy.create_placement_groups(resource_pools=pools)

    assert train_node_count == 2
    train_locations = strategy._worker_locations["train"]
    assert len(train_locations) == 16

    train_pg_keys = {loc[0] for loc in train_locations}
    assert len(train_pg_keys) == 2

    node_0_locs = [loc for loc in train_locations if "node_0" in loc[0]]
    node_1_locs = [loc for loc in train_locations if "node_1" in loc[0]]
    assert len(node_0_locs) == 8
    assert len(node_1_locs) == 8

    assert sorted(loc[1] for loc in node_0_locs) == list(range(8))
    assert sorted(loc[1] for loc in node_1_locs) == list(range(8))

    buffer_locations = strategy._worker_locations["buffer"]
    assert len(buffer_locations) == 2

    mapping = strategy.get_storage_to_train_workers()
    assert len(mapping) == 2
    assert mapping[0] == list(range(0, 8))
    assert mapping[1] == list(range(8, 16))
