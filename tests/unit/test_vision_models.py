from typing import Tuple

import pytest
import torch

from rlightning.models.vision.cnn import NatureCNN

@pytest.mark.parametrize("image_shape", [(84, 84), (210, 160)])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("image_format", ["HWC", "CHW"])
def test_any_shape_as_input(image_shape: Tuple[int, int], batch_size: int, image_format: str):
    model = NatureCNN(image_format=image_format)

    if image_format == "HWC":
        observation = torch.rand((batch_size,) + image_shape + (model.in_channels,))
    else:
        observation = torch.rand((batch_size, model.in_channels) + image_shape)

    features = model(obs=observation)

    assert features.shape == (batch_size, model.out_feature_dim)
