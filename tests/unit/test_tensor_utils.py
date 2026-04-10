import numpy as np
import pytest
import torch

from rlightning.utils.utils import InternalFlag, to_device, to_numpy, torch_dtype_from_precision


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("bf16", torch.bfloat16),
        ("bf16-mixed", torch.bfloat16),
        (16, torch.float16),
        ("16", torch.float16),
        ("fp16", torch.float16),
        ("16-mixed", torch.float16),
        (32, torch.float32),
        ("32", torch.float32),
        ("32-true", torch.float32),
        (None, None),
    ],
)
def test_torch_dtype_from_precision_variants(precision, expected_dtype):
    assert torch_dtype_from_precision(precision) is expected_dtype


def test_torch_dtype_from_precision_rejects_unknown_value():
    with pytest.raises(ValueError, match="Could not parse the precision"):
        torch_dtype_from_precision("fp8")


def test_bfloat16_numpy_round_trip_preserves_values():
    source = {"weights": torch.tensor([1.5, -2.0], dtype=torch.bfloat16)}

    numpy_data = to_numpy(source)
    restored = to_device(numpy_data, "cpu")

    assert isinstance(numpy_data["weights"], np.ndarray)
    assert numpy_data["weights"].dtype == np.uint16
    assert restored["weights"].dtype == torch.bfloat16
    assert torch.equal(restored["weights"], source["weights"])


def test_to_numpy_rejects_raw_uint16_tensors():
    with pytest.raises(ValueError, match="haven't support converting uint16 tensor"):
        to_numpy(torch.tensor([1, 2], dtype=torch.uint16))


def test_internal_flag_get_env_vars_reflects_environment(monkeypatch):
    monkeypatch.setenv("RLIGHTNING_DEBUG", "1")
    monkeypatch.setenv("RLIGHTNING_VERBOSE", "0")
    monkeypatch.setenv("RLIGHTNING_REMOTE_TRAIN", "1")
    monkeypatch.setenv("RLIGHTNING_REMOTE_EVAL", "0")
    monkeypatch.setenv("RLIGHTNING_REMOTE_STORAGE", "1")
    monkeypatch.setenv("RLIGHTNING_REMOTE_ENV", "0")

    assert InternalFlag.get_env_vars() == {
        "RLIGHTNING_DEBUG": "1",
        "RLIGHTNING_VERBOSE": "0",
        "RLIGHTNING_REMOTE_TRAIN": "1",
        "RLIGHTNING_REMOTE_EVAL": "0",
        "RLIGHTNING_REMOTE_STORAGE": "1",
        "RLIGHTNING_REMOTE_ENV": "0",
    }
