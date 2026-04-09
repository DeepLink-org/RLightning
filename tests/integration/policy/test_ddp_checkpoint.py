import pytest
import torch

from rlightning.utils.config import (
    ClusterConfig,
    Config,
    PolicyConfig,
    TrainConfig,
    WeightBufferConfig,
)
from rlightning.utils.utils import InternalFlag
from tests.test_utils import (
    get_test_ray_address,
    init_test_ray,
    setup_ray_cluster,
    should_skip_gpu_test,
    should_skip_ray_test,
    teardown_ray_cluster,
)
from tests.tests_utils.ddp_checkpoint_policy import DDPCheckpointPolicy


@pytest.fixture(scope="module", autouse=True)
def _ddp_ray_cluster():
    if should_skip_ray_test():
        pytest.skip("Ray not available")
    if should_skip_gpu_test():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 2:
        pytest.skip("DDP checkpoint test requires at least 2 GPUs")

    started_cluster = False
    import ray

    ray_address = get_test_ray_address(default="local")

    if ray.is_initialized():
        ray.shutdown()

    if ray_address == "local":
        setup_ray_cluster(num_cpus=2, num_gpus=2, local_mode=False)
        started_cluster = True
    else:
        init_test_ray(log_to_driver=False, default_address=ray_address)

    if ray.cluster_resources().get("GPU", 0) < 2:
        pytest.skip("DDP checkpoint test requires at least 2 GPUs in Ray cluster")

    yield
    if started_cluster:
        teardown_ray_cluster()
    elif ray.is_initialized():
        ray.shutdown()


def _load_checkpoint(ckpt_path):
    return torch.load(ckpt_path, map_location="cpu")


def _format_checkpoint(state):
    formatted = {}
    for module_name, module_state in state.items():
        formatted[module_name] = {
            param_name: tensor.detach().cpu().tolist() for param_name, tensor in module_state.items()
        }
    return formatted


@pytest.mark.integration
@pytest.mark.gpu
def test_ddp_checkpoints_consistent(tmp_path, monkeypatch):
    monkeypatch.setenv("RLIGHTNING_REMOTE_TRAIN", "1")
    monkeypatch.setenv("RLIGHTNING_REMOTE_EVAL", "0")
    assert InternalFlag.REMOTE_TRAIN is True

    policy_cfg = PolicyConfig(
        type="DDPCheckpointPolicy",
        rollout_mode="sync",
        optim_cfg=Config.from_dict({"lr": 1e-2}),
        weight_buffer=WeightBufferConfig(buffer_strategy="None"),
    )
    train_config = TrainConfig(max_epochs=1, parallel="ddp")

    from rlightning.utils.builders import build_policy_group

    policy_group = build_policy_group(
        policy_cls=policy_cfg.type,
        policy_cfg=policy_cfg,
        cluster_cfg=ClusterConfig(train_worker_num=2, eval_worker_num=0),
    )

    policy_group.init_train(train_config, env_meta=None)

    initial_paths = []
    initial_refs = []
    for idx, policy in enumerate(policy_group.train_list):
        ckpt_path = tmp_path / f"initial_rank_{idx}_epoch_0.pth"
        initial_paths.append(ckpt_path)
        initial_refs.append(policy.save_checkpoint.remote(ckpt_path))

    import ray

    ray.get(initial_refs)

    initial_states = [_load_checkpoint(ckpt_path) for ckpt_path in initial_paths]
    # print("initial checkpoints:", [_format_checkpoint(state) for state in initial_states])

    data = {
        "x": torch.randn(16, 4),
        "y": torch.randn(16, 2),
    }
    policy_group.update_dataset([data, data])

    for _ in range(3):
        policy_group.train()

    save_paths = []
    save_refs = []
    for idx, policy in enumerate(policy_group.train_list):
        ckpt_path = tmp_path / f"rank_{idx}.pth"
        save_paths.append(ckpt_path)
        save_refs.append(policy.save_checkpoint.remote(ckpt_path))

    ray.get(save_refs)

    states = [_load_checkpoint(ckpt_path) for ckpt_path in save_paths]
    # print("after 3 steps checkpoints:", [_format_checkpoint(state) for state in states])
    assert states[0].keys() == states[1].keys()
    for module_name in states[0]:
        module_state_0 = states[0][module_name]
        module_state_1 = states[1][module_name]
        assert module_state_0.keys() == module_state_1.keys()
        for param_name in module_state_0:
            torch.testing.assert_close(module_state_0[param_name], module_state_1[param_name])
    for module_name in states[0]:
        module_state_0 = states[0][module_name]
        module_state_1 = states[1][module_name]
        assert module_state_0.keys() == module_state_1.keys()
        for param_name in module_state_0:
            torch.testing.assert_close(module_state_0[param_name], module_state_1[param_name])
