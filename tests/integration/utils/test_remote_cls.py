import logging
import os

import pytest
import ray
import torch

from tests.test_utils import init_test_ray
from tests.tests_utils.remote_class_helper import Custom

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.fixture(autouse=True)
def _test_ray_runtime():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    python_path = os.environ.get("PYTHONPATH", "")
    runtime_env = {
        "env_vars": {
            "RAY_DEBUG": "1",
            "PYTHONPATH": python_path,
        },
    }
    try:
        init_test_ray(num_gpus=1, local_mode=False, runtime_env=runtime_env)
    except Exception as exc:
        logging.error(f"Failed to initialize Ray for test: {exc}")
        pytest.fail(f"Failed to initialize Ray for test: {exc}")

    logging.info("Connected to test Ray cluster.")

    yield

    if ray.is_initialized():
        ray.shutdown()


def test_remote():
    remote_cls = Custom.as_remote()

    instance = remote_cls.options(num_gpus=0.5).remote()

    res = ray.get(instance._get_gpu_ids.remote())
    print(".... gpu ids:", res)
    assert isinstance(res, list) and len(res) == 1
