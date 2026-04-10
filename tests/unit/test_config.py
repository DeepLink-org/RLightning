import pytest
from omegaconf import OmegaConf

from rlightning.utils.config import (
    BufferConfig,
    Config,
    EnvConfig,
    LogConfig,
    MainConfig,
    PolicyConfig,
    TrainConfig,
    WeightBufferConfig,
    validate_config_for_placement,
)
import rlightning.utils.placement as placement_module


def test_buffer_config_sets_default_sampler_by_buffer_type():
    rollout_cfg = BufferConfig(type="RolloutBuffer", capacity=8)
    replay_cfg = BufferConfig(type="ReplayBuffer", capacity=8)

    assert rollout_cfg.sampler.type == "all"
    assert replay_cfg.sampler.type == "uniform"


def test_weight_buffer_config_rejects_shared_strategy_for_non_cpu_weight_buffer():
    with pytest.raises(ValueError, match="Shared buffer strategy"):
        WeightBufferConfig(type="WeightBuffer", buffer_strategy="Shared")


def test_env_config_rejects_vector_env_for_unsupported_backend():
    with pytest.raises(ValueError, match="Vectorized environments are only supported"):
        EnvConfig(
            name="bad-env",
            backend="mujoco",
            task="cartpole",
            num_envs=2,
        )


def test_env_config_sets_maniskill_control_mode():
    cfg = EnvConfig(
        name="maniskill-env",
        backend="maniskill",
        task="pick_cube",
        init_params={},
        policy_setup="widowx",
    )

    assert cfg.init_params.control_mode == "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"


def test_main_config_from_dict_formats_validation_errors(make_main_config_dict):
    invalid_config = make_main_config_dict()
    del invalid_config["train"]["max_epochs"]

    with pytest.raises(ValueError, match="field 'train.max_epochs'"):
        MainConfig.from_dict(invalid_config)


def test_main_config_from_omegaconf_promotes_nested_config_types(make_main_config_dict):
    cfg = MainConfig.from_omegaconf(OmegaConf.create(make_main_config_dict()))

    assert isinstance(cfg.env, list)
    assert isinstance(cfg.env[0], EnvConfig)
    assert isinstance(cfg.buffer, BufferConfig)
    assert isinstance(cfg.policy, PolicyConfig)
    assert isinstance(cfg.train, TrainConfig)
    assert isinstance(cfg.log, LogConfig)


def test_main_config_converts_single_env_config_to_list(make_main_config_dict):
    config_dict = make_main_config_dict()
    config_dict["env"] = config_dict["env"][0]

    config = MainConfig.from_dict(config_dict)

    assert isinstance(config.env, list)
    assert len(config.env) == 1
    assert isinstance(config.env[0], EnvConfig)


def test_config_allows_extra_fields_and_recursively_wraps_extra_dicts(make_main_config_dict):
    config_dict = make_main_config_dict()
    config_dict["extra_level_1"] = {"level_2": {"param": 42}}

    config = MainConfig.from_dict(config_dict)

    assert isinstance(config.extra_level_1, Config)
    assert isinstance(config.extra_level_1.level_2, Config)
    assert config.extra_level_1.level_2.param == 42


def test_config_get_and_getitem_return_field_values():
    cfg = Config.from_dict({"alpha": 1, "nested": {"beta": 2}})

    assert cfg.get("alpha") == 1
    assert cfg["alpha"] == 1
    assert cfg.get("missing", "fallback") == "fallback"
    assert isinstance(cfg.nested, Config)
    assert cfg.nested.beta == 2


def test_main_config_to_dict_and_yaml_include_nested_fields(make_main_config_dict):
    config = MainConfig.from_dict(make_main_config_dict())

    as_dict = config.to_dict()
    as_yaml = config.to_yaml()

    assert as_dict["buffer"]["type"] == "RolloutBuffer"
    assert as_dict["policy"]["type"] == "SimplePPOPolicy"
    assert "buffer:" in as_yaml
    assert "policy:" in as_yaml


def test_validate_config_for_placement_applies_colocate_overrides(make_main_config_dict, monkeypatch):
    class FakeResourceManager:
        def get_placement_strategy(self) -> str:
            return "colocate"

    monkeypatch.setattr(placement_module, "get_global_resource_manager", lambda: FakeResourceManager())

    config = MainConfig.from_dict(make_main_config_dict())
    validated = validate_config_for_placement(config)

    assert validated.cluster.train_each_gpu_num == 0.1
    assert validated.cluster.eval_each_gpu_num == 0.1
    assert validated.cluster.is_colocated is True
    assert validated.policy.weight_buffer.buffer_strategy == "None"
    assert validated.env[0].num_gpus == 0.1
    assert validated.env[0].num_cpus == 1
