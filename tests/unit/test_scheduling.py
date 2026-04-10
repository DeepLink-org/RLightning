import pytest

from rlightning.utils.config import MainConfig
from rlightning.utils.placement.scheduling import ComponentScheduling, Scheduling, setup_component_scheduling


def test_scheduling_reports_totals_and_dict():
    scheduling = Scheduling(worker_num=4, num_cpus=2, num_gpus=1.0)

    assert scheduling.total_gpus() == 4.0
    assert scheduling.total_cpus() == 8
    assert scheduling.to_dict() == {
        "worker_num": 4,
        "num_cpus": 2,
        "num_gpus": 1.0,
        "node_list": None,
    }


def test_component_scheduling_reports_pool_requirements():
    scheduling = ComponentScheduling(
        env_worker=[
            Scheduling(worker_num=2, num_cpus=4, num_gpus=0.5),
            Scheduling(worker_num=1, num_cpus=2, num_gpus=0.0),
        ],
        train_worker=Scheduling(worker_num=2, num_cpus=1, num_gpus=1.0),
        eval_worker=Scheduling(worker_num=3, num_cpus=1, num_gpus=0.25),
        buffer_worker=Scheduling(worker_num=1, num_cpus=1, num_gpus=0.0),
    )

    assert scheduling.train_pool_requirements() == (2.0, 3)
    assert scheduling.rollout_pool_requirements() == (1.75, 13)
    assert scheduling.get_component_requirements("env") == (1.0, 10)


def test_component_scheduling_rejects_unknown_component_type():
    scheduling = ComponentScheduling()

    with pytest.raises(ValueError, match="Invalid component type"):
        scheduling.get_component_requirements("learner")


def test_component_scheduling_returns_zero_for_missing_workers():
    scheduling = ComponentScheduling()

    assert scheduling.get_component_requirements("train") == (0, 0)
    assert scheduling.get_component_requirements("eval") == (0, 0)
    assert scheduling.get_component_requirements("env") == (0, 0)
    assert scheduling.get_component_requirements("buffer") == (0, 0)


def test_infer_auto_buffer_worker_num_uses_train_gpu_count_and_node_gpu_capacity():
    scheduling = ComponentScheduling(
        train_worker=Scheduling(worker_num=5, num_cpus=1, num_gpus=1.0),
        buffer_worker=Scheduling(worker_num="auto", num_cpus=1, num_gpus=0.0),
    )

    scheduling.infer_auto_buffer_worker_num(
        {
            "node_id_to_resources": {
                "node-a": {"GPU": 4},
                "node-b": {"GPU": 4},
            }
        }
    )

    assert scheduling.buffer_worker.worker_num == 2


def test_setup_component_scheduling_forces_single_buffer_worker_for_unified_storage(make_main_config_dict):
    config = MainConfig.from_dict(
        make_main_config_dict(
            cluster={"buffer_worker_num": 4},
            buffer={"storage": {"type": "unified", "device": "cpu"}},
        )
    )

    scheduling = setup_component_scheduling(config)

    assert scheduling.buffer_worker.worker_num == 1
    assert config.cluster.buffer_worker_num == 1
