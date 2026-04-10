"""Launch test-only minimal experiments through the normal RLightning entrypoint."""

from __future__ import annotations

from pathlib import Path

import torch

from rlightning.utils.builders import build_data_buffer, build_engine, build_env_group, build_policy_group
from rlightning.utils.config import MainConfig
from rlightning.utils.launch import launch


def main(config: MainConfig) -> None:
    env_group = build_env_group(config.env)
    policy_group = build_policy_group(
        policy_cls=config.policy.type,
        policy_cfg=config.policy,
        cluster_cfg=config.cluster,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    )
    buffer = build_data_buffer(
        buffer_cls=config.buffer.type,
        buffer_cfg=config.buffer,
    )
    engine = build_engine(
        config=config,
        env_group=env_group,
        policy_group=policy_group,
        buffer=buffer,
    )

    try:
        engine.run()
    finally:
        policy_group.shutdown()
        env_group.close()


if __name__ == "__main__":
    launch(main_func=main, config_path=Path(__file__).parent / "mini_experiment_conf")
