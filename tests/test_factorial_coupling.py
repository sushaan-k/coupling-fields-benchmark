import numpy as np
import pytest

from mapreg.factorial_coupling import (
    FactorialCouplingRefusal,
    _mismatched_joint,
    factorial_association_contrast,
    factorial_observed_law,
    fit_factorial_coupling,
)


def _balanced(log_field: np.ndarray) -> np.ndarray:
    table = np.exp(log_field - np.max(log_field))
    for _ in range(100):
        table /= table.sum(axis=1, keepdims=True) * table.shape[0]
        table /= table.sum(axis=0, keepdims=True) * table.shape[1]
    return table / table.sum()


def _joint_panel() -> np.ndarray:
    fields = (
        np.array([[0.7, -0.2, -0.5], [-0.2, 0.4, -0.2], [-0.5, -0.2, 0.7]]),
        np.array([[0.2, 0.3, -0.5], [-0.4, 0.5, -0.1], [0.2, -0.8, 0.6]]),
        np.array([[0.4, -0.1, -0.3], [-0.2, 0.3, -0.1], [-0.2, -0.2, 0.4]]),
        np.array([[-0.2, 0.5, -0.3], [0.4, -0.3, -0.1], [-0.2, -0.2, 0.4]]),
    )
    return np.stack([_balanced(field) for field in fields]).reshape(2, 2, 3, 3)


def _guide_channel() -> np.ndarray:
    return np.array(
        [
            [[0.90, 0.14], [0.10, 0.86]],
            [[0.84, 0.08], [0.16, 0.92]],
        ]
    )


def _dual_label_channel() -> np.ndarray:
    dual = np.linspace(0.03, 0.15, 9)
    channel = np.zeros((10, 9))
    channel[:9] = np.diag(1.0 - dual)
    channel[9] = dual
    return channel


def _state_dependent_guide_channel() -> np.ndarray:
    channel = np.empty((2, 2, 2, 9))
    for challenge in range(2):
        for state in range(9):
            error_0 = 0.05 + 0.01 * ((state + challenge) % 4)
            error_1 = 0.08 + 0.01 * ((2 * state + challenge) % 3)
            channel[challenge, :, :, state] = np.array(
                [[1.0 - error_0, error_1], [error_0, 1.0 - error_1]]
            )
    return channel


def test_population_recovery_with_guide_swaps_and_dual_label_category() -> None:
    joint = _joint_panel()
    prevalence = np.array([[0.72, 0.38], [0.28, 0.62]])
    guide = _guide_channel()
    eta = np.array([[0.12, 0.20], [0.08, 0.16]])
    observation = _dual_label_channel()
    law = factorial_observed_law(
        joint,
        prevalence,
        guide,
        eta,
        observation_channel=observation,
    )
    counts = np.rint(80_000_000 * law).astype(int)

    fit = fit_factorial_coupling(
        counts,
        guide,
        (3, 3),
        eta,
        observation_channel=observation,
    )

    np.testing.assert_allclose(fit.joint, joint, atol=4e-5)
    np.testing.assert_allclose(fit.nuisance_prevalence, prevalence, atol=2e-5)
    np.testing.assert_allclose(fit.expected_law, law, atol=2e-5)
    expected_amplification = 1.0 / (
        fit.diagnostics.guide_min_singular[None, :]
        * fit.diagnostics.observation_min_singular
        * (1.0 - eta)
        * fit.nuisance_prevalence
    )
    np.testing.assert_allclose(
        fit.diagnostics.coarse_inverse_amplification, expected_amplification
    )
    np.testing.assert_allclose(fit.diagnostics.correctly_linked_fraction, 1.0 - eta)
    assert fit.expected_law[:, :, -1].sum() > 0.05
    assert fit.diagnostics.gradient_norm < 1e-6


def test_population_recovery_with_state_dependent_guide_errors() -> None:
    joint = _joint_panel()
    prevalence = np.array([[0.63, 0.41], [0.37, 0.59]])
    guide = _state_dependent_guide_channel()
    law = factorial_observed_law(joint, prevalence, guide, 0.0)
    counts = np.rint(100_000_000 * law).astype(int)

    fit = fit_factorial_coupling(counts, guide, (3, 3), 0.0)

    np.testing.assert_allclose(fit.joint, joint, atol=5e-5)
    np.testing.assert_allclose(fit.nuisance_prevalence, prevalence, atol=2e-5)
    np.testing.assert_allclose(fit.expected_law, law, atol=2e-5)


def test_one_rank_deficient_state_specific_guide_channel_refuses() -> None:
    guide = _state_dependent_guide_channel()
    guide[:, :, :, 4] = 0.5
    counts = np.full((2, 2, 9), 1000)

    with pytest.raises(FactorialCouplingRefusal, match="rank deficient"):
        fit_factorial_coupling(counts, guide, (3, 3), 0.0)


def test_guide_mixing_is_not_within_arm_pair_mismatch() -> None:
    joint = _joint_panel()
    prevalence = np.array([[0.75, 0.75], [0.25, 0.25]])
    guide = _guide_channel()
    identity_guide = np.broadcast_to(np.eye(2), guide.shape)
    no_mismatch = factorial_observed_law(joint, prevalence, guide, 0.0)
    heavy_mismatch = factorial_observed_law(joint, prevalence, guide, 0.65)
    no_guide_mixing = factorial_observed_law(joint, prevalence, identity_guide, 0.65)

    np.testing.assert_allclose(
        no_mismatch.sum(axis=2), heavy_mismatch.sum(axis=2), atol=1e-12
    )
    assert not np.allclose(
        heavy_mismatch.sum(axis=2), no_guide_mixing.sum(axis=2), atol=1e-3
    )


def test_random_within_arm_mismatch_preserves_nonuniform_margins() -> None:
    joint = np.array(
        [[0.31, 0.08, 0.01], [0.06, 0.19, 0.05], [0.03, 0.07, 0.20]]
    )
    observed = _mismatched_joint(joint, 0.37)
    recovered = (
        observed - 0.37 * np.outer(observed.sum(axis=1), observed.sum(axis=0))
    ) / 0.63

    np.testing.assert_allclose(observed.sum(axis=0), joint.sum(axis=0))
    np.testing.assert_allclose(observed.sum(axis=1), joint.sum(axis=1))
    np.testing.assert_allclose(recovered, joint)


def test_endpoint_invisible_factorial_interaction_is_recovered() -> None:
    route = np.array(
        [[0.8, -0.4, -0.4], [-0.4, 0.2, 0.2], [-0.4, 0.2, 0.2]]
    )
    uniform = np.full((3, 3), 1.0 / 9.0)
    changed = _balanced(route)
    joint = np.broadcast_to(uniform, (2, 2, 3, 3)).copy()
    joint[0, 1] = changed
    prevalence = np.full((2, 2), 0.5)
    guide = _guide_channel()
    law = factorial_observed_law(joint, prevalence, guide, 0.1)
    fit = fit_factorial_coupling(
        np.rint(50_000_000 * law), guide, (3, 3), 0.1
    )
    contrast = factorial_association_contrast(fit, 0, 1, 1, 0)

    np.testing.assert_allclose(changed.sum(axis=0), uniform.sum(axis=0), atol=1e-12)
    np.testing.assert_allclose(changed.sum(axis=1), uniform.sum(axis=1), atol=1e-12)
    assert np.linalg.norm(contrast) > 1.0
    np.testing.assert_allclose(
        contrast.sum(axis=0), np.zeros(3), atol=1e-10
    )
    np.testing.assert_allclose(
        contrast.sum(axis=1), np.zeros(3), atol=1e-10
    )


def test_rank_conditioning_and_mismatch_amplification_refuse() -> None:
    counts = np.full((2, 2, 9), 1000)
    identity_guide = np.broadcast_to(np.eye(2), (2, 2, 2))
    rank_deficient = np.broadcast_to(
        np.array([[0.5, 0.5], [0.5, 0.5]]), (2, 2, 2)
    )
    with pytest.raises(FactorialCouplingRefusal, match="rank deficient"):
        fit_factorial_coupling(counts, rank_deficient, (3, 3), 0.0)
    with pytest.raises(FactorialCouplingRefusal, match="rank deficient"):
        fit_factorial_coupling(
            counts,
            identity_guide,
            (3, 3),
            0.0,
            observation_channel=np.full((9, 9), 1.0 / 9.0),
        )

    poorly_conditioned = np.broadcast_to(
        np.array([[0.51, 0.49], [0.49, 0.51]]), (2, 2, 2)
    )
    with pytest.raises(FactorialCouplingRefusal, match="poorly conditioned"):
        fit_factorial_coupling(counts, poorly_conditioned, (3, 3), 0.0)

    with pytest.raises(FactorialCouplingRefusal, match="inverse screening"):
        fit_factorial_coupling(
            counts,
            identity_guide,
            (3, 3),
            0.96,
            min_correctly_linked_fraction=0.01,
            max_coarse_inverse_amplification=20.0,
        )


def test_retention_arm_and_joint_support_refuse() -> None:
    guide = np.broadcast_to(np.eye(2), (2, 2, 2))
    counts = np.full((2, 2, 9), 20)
    with pytest.raises(FactorialCouplingRefusal, match="correctly linked"):
        fit_factorial_coupling(counts, guide, (3, 3), 0.6)
    with pytest.raises(FactorialCouplingRefusal, match="effective arm support"):
        fit_factorial_coupling(
            counts, guide, (3, 3), 0.0, min_effective_arm_count=250.0
        )
    with pytest.raises(FactorialCouplingRefusal, match="joint-cell support"):
        fit_factorial_coupling(
            counts,
            guide,
            (3, 3),
            0.0,
            min_effective_arm_count=1.0,
            min_expected_joint_count=25.0,
        )


def test_rare_fitted_arm_refuses_prevalence_amplification() -> None:
    joint = _joint_panel()
    prevalence = np.array([[0.99, 0.99], [0.01, 0.01]])
    guide = np.broadcast_to(np.eye(2), (2, 2, 2))
    law = factorial_observed_law(joint, prevalence, guide, 0.0)
    counts = np.rint(20_000_000 * law).astype(int)

    with pytest.raises(FactorialCouplingRefusal, match="inverse screening"):
        fit_factorial_coupling(
            counts,
            guide,
            (3, 3),
            0.0,
            max_coarse_inverse_amplification=50.0,
        )


def test_impossible_observed_category_refuses() -> None:
    joint = _joint_panel()
    prevalence = np.full((2, 2), 0.5)
    guide = _guide_channel()
    channel = np.vstack((np.eye(9), np.zeros((1, 9))))
    law = factorial_observed_law(
        joint, prevalence, guide, 0.0, observation_channel=channel
    )
    counts = np.rint(1_000_000 * law).astype(int)
    counts[0, 0, -1] = 1
    with pytest.raises(FactorialCouplingRefusal, match="impossible"):
        fit_factorial_coupling(
            counts,
            guide,
            (3, 3),
            0.0,
            observation_channel=channel,
        )


@pytest.mark.parametrize(
    "channel",
    [2.0 * np.eye(9), np.eye(9) - 0.1 * np.ones((9, 9))],
)
def test_forward_law_rejects_invalid_observation_channels(channel: np.ndarray) -> None:
    with pytest.raises(ValueError, match="observation_channel"):
        factorial_observed_law(
            _joint_panel(),
            np.full((2, 2), 0.5),
            _guide_channel(),
            0.0,
            observation_channel=channel,
        )
