import numpy as np
import pytest

from rlightning.buffer.sampler import AllDataSampler, BatchSampler, UniformSampler


def test_uniform_sampler_returns_requested_batch_size_with_valid_indices():
    sampler = UniformSampler()

    indices = sampler.sample(batch_size=20, data_size=5)

    assert indices.shape == (20,)
    assert np.all(indices >= 0)
    assert np.all(indices < 5)


def test_uniform_sampler_can_repeat_indices_when_sampling_with_replacement():
    sampler = UniformSampler()
    np.random.seed(0)

    indices = sampler.sample(batch_size=5, data_size=2)

    assert len(indices) == 5
    assert np.all(indices >= 0)
    assert np.all(indices < 2)
    assert len(set(indices.tolist())) < 5


def test_all_data_sampler_warns_for_partial_batch_and_returns_all_indices():
    sampler = AllDataSampler()

    with pytest.warns(UserWarning, match="sample all data from the buffer"):
        indices = sampler.sample(batch_size=2, data_size=4, shuffle=True)

    assert sorted(indices.tolist()) == [0, 1, 2, 3]


def test_all_data_sampler_returns_empty_indices_for_empty_input():
    sampler = AllDataSampler()

    indices = sampler.sample(batch_size=0, data_size=0, shuffle=False)

    assert indices.size == 0
    assert indices.tolist() == []


def test_batch_sampler_rejects_oversized_batches():
    sampler = BatchSampler()

    with pytest.raises(ValueError, match="batch_size must be less than or equal to data_size"):
        sampler.sample(batch_size=5, data_size=4)


def test_batch_sampler_samples_without_replacement():
    sampler = BatchSampler()

    indices = sampler.sample(batch_size=4, data_size=10)

    assert len(indices) == 4
    assert len(set(indices.tolist())) == 4
