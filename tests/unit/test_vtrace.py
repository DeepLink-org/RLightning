import torch

from rlightning.policy.utils.vtrace import batch_step_correction, vtrace_correction


def test_vtrace_correction_reduces_to_clipped_td_error_when_gamma_is_zero():
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 1.5])
    next_values = torch.tensor([10.0, 20.0])
    dones = torch.tensor([False, True])
    log_rhos = torch.log(torch.tensor([2.0, 0.5]))

    vs, advantages = vtrace_correction(
        rewards=rewards,
        values=values,
        next_values=next_values,
        dones=dones,
        log_rhos=log_rhos,
        gamma=0.0,
        rho_bar=1.0,
        c_bar=1.0,
    )

    clipped_rhos = torch.tensor([1.0, 0.5])
    expected = clipped_rhos * (rewards - values)

    assert torch.allclose(vs, values + expected)
    assert torch.allclose(advantages, expected)


def test_batch_step_correction_reduces_to_clipped_td_error_when_gamma_is_zero():
    rewards = torch.tensor([[1.0, 2.0]])
    values = torch.tensor([[0.5, 1.5]])
    next_values = torch.tensor([[10.0, 20.0]])
    log_rhos = torch.log(torch.tensor([[2.0, 0.5]]))
    dones = torch.tensor([[False, True]])

    target_values, advantages = batch_step_correction(
        rewards=rewards,
        values=values,
        next_values=next_values,
        log_rhos=log_rhos,
        dones=dones,
        gamma=0.0,
        rho_bar=1.0,
        c_bar=1.0,
    )

    clipped_rhos = torch.tensor([[1.0, 0.5]])
    expected = clipped_rhos * (rewards - values)

    assert torch.allclose(target_values, values + expected)
    assert torch.allclose(advantages, expected)
