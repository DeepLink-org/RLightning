"""Unit tests for buffer preprocessors."""

from __future__ import annotations

from typing import Dict

import gymnasium as gym
import numpy as np
import pytest
import torch

from rlightning.buffer.utils.preprocessors import (
    BoxFlattenPreprocessor,
    DiscretePreprocessor,
    NonPreprocessor,
    default_obs_preprocessor,
    default_reward_preprocessor,
    get_preprocessor_cls,
)

def test_preprocessor_call_rejects_invalid_type():
    """Preprocessor __call__ should assert on unsupported input types.
    But origin function uses assert, which will be ignored when python
    is run with -O flag. So maybe the origin function should raise
    TypeError instead. TODO: @yangzhenyu @qiujiawei
    """
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    pre = NonPreprocessor(space)

    with pytest.raises(AssertionError):
        _ = pre(123)


def test_non_preprocessor_returns_input_and_shape():
    """NonPreprocessor should return inputs unchanged and expose shape."""
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2, 3))
    pre = NonPreprocessor(space)
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # simple origin function, no need to test multiple types.

    assert pre.shape == (2, 3)
    assert pre.transform(data) is data
    assert pre.batch_transform(data) is data


def test_box_flatten_preprocessor_transform_and_batch():
    """BoxFlattenPreprocessor should flatten single and batched observations."""
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2, 2))
    pre = BoxFlattenPreprocessor(space)

    assert pre.shape == (4,)

    torch_obs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    flat_torch = pre.transform(torch_obs)
    assert flat_torch.shape == (4,)
    assert flat_torch.tolist() == [1.0, 2.0, 3.0, 4.0]

    np_obs = np.array([[5.0, 6.0], [7.0, 8.0]])
    flat_np = pre.transform(np_obs)
    assert flat_np.shape == (4,)
    assert flat_np.tolist() == [5.0, 6.0, 7.0, 8.0]

    batched = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    flat_batch = pre.batch_transform(batched)
    assert flat_batch.shape == (2, 4)
    assert flat_batch.tolist() == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]

    tuple_batch = (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[9.0, 10.0], [11.0, 12.0]]))
    tuple_out = pre.batch_transform(tuple_batch)
    assert isinstance(tuple_out, tuple)
    assert tuple_out[0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert tuple_out[1].tolist() == [9.0, 10.0, 11.0, 12.0]


def test_discrete_preprocessor_one_hot_and_batch_tuple():
    """DiscretePreprocessor should one-hot encode inputs and support tuple batches."""
    space = gym.spaces.Discrete(3)
    pre = DiscretePreprocessor(space)

    assert pre.shape == (3,)

    np_obs = np.array(1)
    one_hot_np = pre.transform(np_obs)
    assert isinstance(one_hot_np, np.ndarray)
    assert one_hot_np.tolist() == [0.0, 1.0, 0.0]

    torch_obs = torch.tensor(2)
    one_hot_torch = pre.transform(torch_obs)
    assert isinstance(one_hot_torch, torch.Tensor)
    assert one_hot_torch.tolist() == [0.0, 0.0, 1.0]

    tuple_batch = (np.array(0), np.array(2))
    tuple_out = pre.batch_transform(tuple_batch)
    assert isinstance(tuple_out, tuple)
    assert tuple_out[0].tolist() == [1.0, 0.0, 0.0]
    assert tuple_out[1].tolist() == [0.0, 0.0, 1.0]

    tensor_batch = torch.tensor([0, 2])
    tensor_batch_out = pre.batch_transform(tensor_batch)
    assert isinstance(tensor_batch_out, torch.Tensor)
    assert tensor_batch_out.shape == (2, 3)
    assert tensor_batch_out.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


def test_default_preprocessors_return_input_sequences():
    """default_obs_preprocessor and default_reward_preprocessor should pass through."""
    obs_seq: Dict[str, int] = {"a": 1}
    rew_seq = [1.0, 2.0]

    assert default_obs_preprocessor(obs_seq) is obs_seq
    assert default_reward_preprocessor(rew_seq) is rew_seq


def test_get_preprocessor_cls_selects_by_space_type():
    """get_preprocessor_cls should select the right preprocessor class."""
    box_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    discrete_space = gym.spaces.Discrete(5)
    other_space = gym.spaces.MultiDiscrete([2, 3])

    assert get_preprocessor_cls(box_space) is BoxFlattenPreprocessor
    assert get_preprocessor_cls(discrete_space) is DiscretePreprocessor
    assert get_preprocessor_cls(other_space) is NonPreprocessor
