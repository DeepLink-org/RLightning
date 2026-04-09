import importlib
import inspect
import sys

import pytest

from rlightning.policy.base_policy import PolicyRole
from rlightning.utils.config import PolicyConfig


@pytest.mark.parametrize(
    ("module_path", "class_name"),
    [
        ("rlightning.policy.simple_ppo_policy.ppo_policy", "SimplePPOPolicy"),
        ("rlightning.policy.rsl_rl_policy.rsl_rl_policy", "RSLRLPolicy"),
        ("rlightning.policy.vla_policy.ppo_policy", "VLAPPOPolicy"),
        ("rlightning.policy.supervised_policy", "SimpleSupervisedPolicy"),
    ],
)
def test_policy_is_concrete(module_path: str, class_name: str):
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        pytest.skip(f"Skipping {module_path} import: {exc}")

    policy_cls = getattr(module, class_name, None)
    assert policy_cls is not None, f"Missing class {class_name} in {module_path}"
    assert inspect.isabstract(policy_cls) is False


def test_rsl_rl_policy_raises_clear_error_when_dependency_missing(monkeypatch):
    module = importlib.import_module("rlightning.policy.rsl_rl_policy.rsl_rl_policy")
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rsl_rl":
            raise ImportError("missing rsl_rl")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "rsl_rl", raising=False)
    monkeypatch.setattr("builtins.__import__", fake_import)

    config = PolicyConfig.from_dict(
        {
            "type": "RSLRLPolicy",
            "rollout_mode": "sync",
            "weight_buffer": {"type": "WeightBuffer", "buffer_strategy": "None"},
            "policy_kwargs": {"algorithm": "PPO"},
        }
    )

    with pytest.raises(ImportError, match="rsl_rl\\.algorithms"):
        module.RSLRLPolicy(config, PolicyRole.TRAIN)
