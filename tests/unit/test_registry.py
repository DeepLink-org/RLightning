import pytest

from rlightning.utils.registry.registry import Registry


def test_registry_registers_and_retrieves_default_name():
    registry = Registry("models")

    @registry.register()
    class ToyModel:
        pass

    assert registry.get("ToyModel") is ToyModel
    assert "ToyModel" in registry.module_dict


def test_registry_rejects_duplicate_registration():
    registry = Registry("models")

    @registry.register("toy")
    class ToyModel:
        pass

    with pytest.raises(KeyError, match="already registered"):
        registry.register("toy")(ToyModel)


def test_registry_raises_for_missing_key():
    registry = Registry("models")

    with pytest.raises(KeyError, match="missing_model is not in the models registry"):
        registry.get("missing_model")
