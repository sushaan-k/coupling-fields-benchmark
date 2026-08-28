"""Guide-aware factorial coupling estimation for linked molecular cohorts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


class FactorialCouplingRefusal(ValueError):
    """Raised when the declared factorial coupling is not stably identified."""


@dataclass(frozen=True)
class FactorialCouplingDiagnostics:
    """Likelihood and identification diagnostics for a fitted factorial model."""

    negative_log_likelihood: float
    gradient_norm: float
    iterations: int
    guide_min_singular: np.ndarray
    observation_min_singular: np.ndarray
    coarse_inverse_amplification: np.ndarray
    correctly_linked_fraction: np.ndarray
    effective_linked_arm_count: np.ndarray
    minimum_expected_joint_count: np.ndarray


@dataclass(frozen=True)
class FactorialCouplingFit:
    """Arm-specific joint laws and marginal-free association fields."""

    joint: np.ndarray
    association: np.ndarray
    expected_law: np.ndarray
    nuisance_prevalence: np.ndarray
    diagnostics: FactorialCouplingDiagnostics


def _softmax_free(parameters: np.ndarray) -> np.ndarray:
    logits = np.concatenate((parameters, np.zeros(1, dtype=float)))
    logits -= logits.max()
    probability = np.exp(logits)
    return probability / probability.sum()


def _softmax_free_gradient(probability: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    full = probability * (gradient - float(probability @ gradient))
    return full[:-1]


def _log_association(joint: np.ndarray) -> np.ndarray:
    log_joint = np.log(joint)
    return (
        log_joint
        - log_joint.mean(axis=-2, keepdims=True)
        - log_joint.mean(axis=-1, keepdims=True)
        + log_joint.mean(axis=(-2, -1), keepdims=True)
    )


def _validated_channel(
    channel: np.ndarray,
    name: str,
    min_singular: float,
) -> float:
    _validate_probability_channel(channel, name)
    singular = np.linalg.svd(channel, compute_uv=False)
    smallest = float(singular[-1])
    if np.linalg.matrix_rank(channel) < channel.shape[1]:
        raise FactorialCouplingRefusal(f"{name} is rank deficient")
    if smallest < min_singular:
        raise FactorialCouplingRefusal(f"{name} is too poorly conditioned")
    return smallest


def _validate_probability_channel(channel: np.ndarray, name: str) -> None:
    if channel.ndim != 2 or not np.isfinite(channel).all() or np.any(channel < 0.0):
        raise ValueError(f"{name} must be a finite nonnegative matrix")
    if not np.allclose(channel.sum(axis=0), 1.0, atol=1e-8):
        raise ValueError(f"columns of {name} must sum to one")


def _guide_channels(
    values: np.ndarray,
    challenges: int,
    observed_guides: int,
    true_guides: int,
    observed_categories: int,
) -> np.ndarray:
    channel = np.asarray(values, dtype=float)
    shared_shape = (challenges, observed_guides, true_guides)
    if channel.shape == shared_shape:
        return np.broadcast_to(
            channel[..., None], shared_shape + (observed_categories,)
        ).copy()
    expected_shape = shared_shape + (observed_categories,)
    if channel.shape != expected_shape:
        raise ValueError(
            "guide_channel must have shape (challenge, observed guide, true guide) "
            "or add an observed-state axis"
        )
    return channel.copy()


def _broadcast_arm_values(
    values: float | np.ndarray,
    shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    try:
        result = np.broadcast_to(np.asarray(values, dtype=float), shape).copy()
    except ValueError as error:
        raise ValueError(f"{name} must broadcast to (true guides, challenges)") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _observation_channels(
    values: np.ndarray | None,
    true_guides: int,
    challenges: int,
    observed_categories: int,
    state_pairs: int,
) -> np.ndarray:
    if values is None:
        if observed_categories != state_pairs:
            raise ValueError("identity observation channel requires one category per pair")
        identity = np.eye(state_pairs)
        return np.broadcast_to(
            identity, (true_guides, challenges, state_pairs, state_pairs)
        ).copy()

    channel = np.asarray(values, dtype=float)
    if channel.ndim == 2:
        if channel.shape != (observed_categories, state_pairs):
            raise ValueError("shared observation_channel has incompatible dimensions")
        return np.broadcast_to(
            channel,
            (true_guides, challenges, observed_categories, state_pairs),
        ).copy()
    expected_shape = (
        true_guides,
        challenges,
        observed_categories,
        state_pairs,
    )
    if channel.shape != expected_shape:
        raise ValueError(
            "observation_channel must be shared or indexed by true guide and challenge"
        )
    return channel.copy()


def _mismatched_joint(joint: np.ndarray, mismatch_rate: float) -> np.ndarray:
    product = np.outer(joint.sum(axis=1), joint.sum(axis=0))
    return (1.0 - mismatch_rate) * joint + mismatch_rate * product


def _factorial_observed_law_prepared(
    joint: np.ndarray,
    nuisance_prevalence: np.ndarray,
    guide_channel: np.ndarray,
    mismatch_rate: np.ndarray,
    observation_channel: np.ndarray,
) -> np.ndarray:
    true_guides, challenges = joint.shape[:2]
    observed_guides = guide_channel.shape[1]
    observed_categories = observation_channel.shape[2]
    expected = np.zeros((observed_guides, challenges, observed_categories))
    for p in range(true_guides):
        for c in range(challenges):
            observed_states = observation_channel[p, c] @ _mismatched_joint(
                joint[p, c], mismatch_rate[p, c]
            ).ravel()
            expected[:, c] += (
                guide_channel[c, :, p]
                * nuisance_prevalence[p, c]
                * observed_states[None, :]
            )
    return expected


def factorial_observed_law(
    joint: np.ndarray,
    nuisance_prevalence: np.ndarray,
    guide_channel: np.ndarray,
    mismatch_rate: float | np.ndarray = 0.0,
    *,
    observation_channel: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the normalized observed law for every challenge."""

    q = np.asarray(joint, dtype=float)
    prevalence = np.asarray(nuisance_prevalence, dtype=float)
    guide_input = np.asarray(guide_channel, dtype=float)
    if q.ndim != 4 or min(q.shape[-2:]) < 2:
        raise ValueError("joint must have shape (true guides, challenges, U, V)")
    true_guides, challenges, rows, columns = q.shape
    if prevalence.shape != (true_guides, challenges):
        raise ValueError("nuisance_prevalence has incompatible dimensions")
    if (
        np.any(q < 0.0)
        or not np.allclose(q.sum(axis=(-2, -1)), 1.0)
        or np.any(prevalence < 0.0)
        or not np.allclose(prevalence.sum(axis=0), 1.0)
    ):
        raise ValueError("joint and nuisance_prevalence must be probability laws")
    state_pairs = rows * columns
    observed_categories = (
        state_pairs if observation_channel is None else np.asarray(observation_channel).shape[-2]
    )
    observation = _observation_channels(
        observation_channel,
        true_guides,
        challenges,
        observed_categories,
        state_pairs,
    )
    for p in range(true_guides):
        for c in range(challenges):
            _validate_probability_channel(
                observation[p, c], f"observation_channel[{p},{c}]"
            )
    if guide_input.ndim not in (3, 4):
        raise ValueError("guide_channel has incompatible dimensions")
    observed_guides = guide_input.shape[1]
    guide = _guide_channels(
        guide_input,
        challenges,
        observed_guides,
        true_guides,
        observed_categories,
    )
    for c in range(challenges):
        for z in range(observed_categories):
            _validate_probability_channel(
                guide[c, :, :, z], f"guide_channel[{c},:,:,{z}]"
            )
    eta = _broadcast_arm_values(
        mismatch_rate, (true_guides, challenges), "mismatch_rate"
    )
    if np.any((eta < 0.0) | (eta >= 1.0)):
        raise ValueError("mismatch_rate must lie in [0, 1)")

    return _factorial_observed_law_prepared(q, prevalence, guide, eta, observation)


def fit_factorial_coupling(
    counts: np.ndarray,
    guide_channel: np.ndarray,
    state_shape: tuple[int, int],
    mismatch_rate: float | np.ndarray = 0.0,
    *,
    observation_channel: np.ndarray | None = None,
    min_singular: float = 0.25,
    max_coarse_inverse_amplification: float = 100.0,
    min_correctly_linked_fraction: float = 0.5,
    min_effective_arm_count: float = 50.0,
    min_expected_joint_count: float = 2.0,
    maximum_iterations: int = 3_000,
) -> FactorialCouplingFit:
    """Jointly fit guide prevalence and arm-specific observed-state laws.

    Counts are multinomial within each challenge over observed guide and
    observed molecular category. A four-dimensional ``guide_channel`` may vary
    with the observed state category; a three-dimensional channel is shared
    across states. ``mismatch_rate`` is reserved for genuine random within-arm
    U/V record mismatches and defaults to zero. The identity observation
    channel is primary. A supplied joint observation channel is a calibrated
    sensitivity model, not an additional fitted layer.
    """

    table = np.asarray(counts, dtype=float)
    guide_input = np.asarray(guide_channel, dtype=float)
    if (
        table.ndim != 3
        or not np.isfinite(table).all()
        or np.any(table < 0.0)
        or np.any(table.sum(axis=(0, 2)) <= 0.0)
    ):
        raise ValueError("counts must be finite nonnegative (guide, challenge, category) counts")
    if guide_input.ndim not in (3, 4):
        raise ValueError("guide_channel must have three or four dimensions")
    observed_guides, challenges, observed_categories = table.shape
    if guide_input.shape[:2] != (challenges, observed_guides):
        raise ValueError("guide_channel does not match counts")
    true_guides = guide_input.shape[2]
    guide = _guide_channels(
        guide_input,
        challenges,
        observed_guides,
        true_guides,
        observed_categories,
    )
    rows, columns = int(state_shape[0]), int(state_shape[1])
    if rows < 2 or columns < 2:
        raise ValueError("state_shape must contain at least two states per cohort")

    threshold = float(min_singular)
    max_amplification = float(max_coarse_inverse_amplification)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("min_singular must be finite and positive")
    if not np.isfinite(max_amplification) or max_amplification <= 1.0:
        raise ValueError("max_coarse_inverse_amplification must exceed one")

    guide_min = np.empty(challenges)
    for c in range(challenges):
        state_min = [
            _validated_channel(
                guide[c, :, :, z], f"guide_channel[{c},:,:,{z}]", threshold
            )
            for z in range(observed_categories)
        ]
        guide_min[c] = min(state_min)

    state_pairs = rows * columns
    observation = _observation_channels(
        observation_channel,
        true_guides,
        challenges,
        observed_categories,
        state_pairs,
    )
    observation_min = np.empty((true_guides, challenges))
    for p in range(true_guides):
        for c in range(challenges):
            observation_min[p, c] = _validated_channel(
                observation[p, c], f"observation_channel[{p},{c}]", threshold
            )

    eta = _broadcast_arm_values(
        mismatch_rate, (true_guides, challenges), "mismatch_rate"
    )
    if np.any((eta < 0.0) | (eta >= 1.0)):
        raise ValueError("mismatch_rate must lie in [0, 1)")
    correctly_linked = 1.0 - eta
    if np.any(correctly_linked < min_correctly_linked_fraction):
        raise FactorialCouplingRefusal("correctly linked fraction is below threshold")

    base_amplification = np.empty((true_guides, challenges))
    for p in range(true_guides):
        base_amplification[p] = 1.0 / (
            guide_min * observation_min[p] * (1.0 - eta[p])
        )
    if np.any(base_amplification > max_amplification):
        raise FactorialCouplingRefusal("coarse inverse screening bound is too high")

    possible = np.zeros_like(table, dtype=bool)
    for c in range(challenges):
        for p in range(true_guides):
            possible[:, c] |= (
                guide[c, :, p] > 0.0
            ) & (observation[p, c].sum(axis=1)[None, :] > 0.0)
    if np.any((table > 0.0) & ~possible):
        raise FactorialCouplingRefusal("counts occupy an impossible observed category")

    totals = table.sum(axis=(0, 2))
    observed_law = table / totals[None, :, None]
    initial_prevalence = np.empty((true_guides, challenges))
    initial_joint = np.empty((true_guides, challenges, rows, columns))
    for c in range(challenges):
        arm_category = np.empty((true_guides, observed_categories))
        for z in range(observed_categories):
            arm_category[:, z] = (
                np.linalg.pinv(guide[c, :, :, z]) @ observed_law[:, c, z]
            )
        arm_category = np.maximum(arm_category, 1e-8)
        initial_prevalence[:, c] = arm_category.sum(axis=1)
        initial_prevalence[:, c] /= initial_prevalence[:, c].sum()
        for p in range(true_guides):
            observed_state = arm_category[p] / arm_category[p].sum()
            linked = np.linalg.pinv(observation[p, c]) @ observed_state
            linked = linked.reshape(rows, columns)
            linked = (
                linked
                - eta[p, c] * np.outer(linked.sum(axis=1), linked.sum(axis=0))
            ) / (1.0 - eta[p, c])
            linked = np.maximum(linked, 1e-8)
            initial_joint[p, c] = linked / linked.sum()

    prevalence_parameters = np.empty((challenges, true_guides - 1))
    for c in range(challenges):
        logits = np.log(initial_prevalence[:, c])
        prevalence_parameters[c] = logits[:-1] - logits[-1]
    joint_parameters = np.empty((true_guides, challenges, state_pairs - 1))
    for p in range(true_guides):
        for c in range(challenges):
            logits = np.log(initial_joint[p, c].ravel())
            joint_parameters[p, c] = logits[:-1] - logits[-1]
    prevalence_size = prevalence_parameters.size
    initial = np.concatenate((prevalence_parameters.ravel(), joint_parameters.ravel()))
    total_count = float(totals.sum())

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prevalence_free = parameters[:prevalence_size].reshape(
            challenges, true_guides - 1
        )
        joint_free = parameters[prevalence_size:].reshape(
            true_guides, challenges, state_pairs - 1
        )
        prevalence = np.empty((true_guides, challenges))
        joint = np.empty((true_guides, challenges, rows, columns))
        for c in range(challenges):
            prevalence[:, c] = _softmax_free(prevalence_free[c])
        for p in range(true_guides):
            for c in range(challenges):
                joint[p, c] = _softmax_free(joint_free[p, c]).reshape(rows, columns)
        return prevalence, joint

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        prevalence, joint = unpack(parameters)
        expected = _factorial_observed_law_prepared(
            joint, prevalence, guide, eta, observation
        )
        positive = expected > 0.0
        if np.any((table > 0.0) & ~positive):
            return np.inf, np.full_like(parameters, np.nan)
        loss = -float(np.sum(table[positive] * np.log(expected[positive]))) / total_count
        derivative = np.zeros_like(expected)
        derivative[positive] = -table[positive] / (
            total_count * expected[positive]
        )

        prevalence_gradient = np.empty((challenges, true_guides - 1))
        joint_gradient = np.empty((true_guides, challenges, state_pairs - 1))
        for c in range(challenges):
            prevalence_full_gradient = np.empty(true_guides)
            for p in range(true_guides):
                state_law = _mismatched_joint(joint[p, c], eta[p, c])
                observed_state = observation[p, c] @ state_law.ravel()
                arm_derivative = np.einsum(
                    "oz,oz->z", guide[c, :, p], derivative[:, c], optimize=True
                )
                prevalence_full_gradient[p] = arm_derivative @ observed_state

                observed_gradient = prevalence[p, c] * arm_derivative
                latent_gradient = (observation[p, c].T @ observed_gradient).reshape(
                    rows, columns
                )
                row_margin = joint[p, c].sum(axis=1)
                column_margin = joint[p, c].sum(axis=0)
                row_term = latent_gradient @ column_margin
                column_term = row_margin @ latent_gradient
                full_joint_gradient = (1.0 - eta[p, c]) * latent_gradient
                full_joint_gradient += eta[p, c] * (
                    row_term[:, None] + column_term[None, :]
                )
                joint_gradient[p, c] = _softmax_free_gradient(
                    joint[p, c].ravel(), full_joint_gradient.ravel()
                )
            prevalence_gradient[c] = _softmax_free_gradient(
                prevalence[:, c], prevalence_full_gradient
            )
        gradient = np.concatenate(
            (prevalence_gradient.ravel(), joint_gradient.ravel())
        )
        return loss, gradient

    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "ftol": 1e-13,
            "gtol": 1e-9,
            "maxiter": int(maximum_iterations),
        },
    )
    value, gradient = objective(fitted.x)
    if (
        not fitted.success
        or not np.isfinite(value)
        or not np.isfinite(gradient).all()
        or np.linalg.norm(gradient) > 1e-5
    ):
        raise FactorialCouplingRefusal("factorial coupling likelihood did not converge")

    prevalence, joint = unpack(fitted.x)
    amplification = base_amplification / prevalence
    if np.any(amplification > max_amplification):
        raise FactorialCouplingRefusal("coarse inverse screening bound is too high")
    effective_arm = correctly_linked * prevalence * totals[None, :]
    if np.any(effective_arm < min_effective_arm_count):
        raise FactorialCouplingRefusal("effective arm support is below threshold")
    expected_joint = effective_arm[:, :, None, None] * joint
    minimum_joint = expected_joint.min(axis=(-2, -1))
    if np.any(minimum_joint < min_expected_joint_count):
        raise FactorialCouplingRefusal("effective joint-cell support is below threshold")

    expected = _factorial_observed_law_prepared(
        joint, prevalence, guide, eta, observation
    )
    diagnostics = FactorialCouplingDiagnostics(
        negative_log_likelihood=float(value),
        gradient_norm=float(np.linalg.norm(gradient)),
        iterations=int(fitted.nit),
        guide_min_singular=guide_min,
        observation_min_singular=observation_min,
        coarse_inverse_amplification=amplification,
        correctly_linked_fraction=correctly_linked,
        effective_linked_arm_count=effective_arm,
        minimum_expected_joint_count=minimum_joint,
    )
    return FactorialCouplingFit(
        joint=joint,
        association=_log_association(joint),
        expected_law=expected,
        nuisance_prevalence=prevalence,
        diagnostics=diagnostics,
    )


def factorial_association_contrast(
    fit: FactorialCouplingFit,
    perturbation: int,
    control: int,
    challenge: int,
    vehicle: int,
) -> np.ndarray:
    """Return the perturbation-by-challenge contrast of association fields."""

    field = fit.association
    if not (
        0 <= perturbation < field.shape[0]
        and 0 <= control < field.shape[0]
        and 0 <= challenge < field.shape[1]
        and 0 <= vehicle < field.shape[1]
    ):
        raise IndexError("factorial contrast index is out of bounds")
    return (field[perturbation, challenge] - field[perturbation, vehicle]) - (
        field[control, challenge] - field[control, vehicle]
    )
