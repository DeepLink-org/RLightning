import importlib
import inspect

import pytest

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
