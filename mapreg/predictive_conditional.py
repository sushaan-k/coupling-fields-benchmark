"""Normal random-effects prediction for a binary fixed-margin table."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, roots_hermitenorm

from .heterogeneity_adaptive_coupling import (
    CouplingEstimationRefusal,
    _integer_binary_margins,
    _log_choose,
)


@dataclass(frozen=True)
class ConditionalMixturePrediction:
    mean_table: np.ndarray
    support: np.ndarray
    probabilities: np.ndarray
    log_probabilities: np.ndarray
    quadrature_order: int
    convergence_error: float

    def count_interval(self, level: float = 0.95) -> tuple[int, int]:
        """Equal-tail interval for the upper-left count, conditional on margins."""
        if not 0 < level < 1:
            raise ValueError("level must lie between zero and one")
        cumulative = np.cumsum(self.probabilities)
        indices = np.searchsorted(cumulative, [(1 - level) / 2, (1 + level) / 2])
        indices = np.minimum(indices, len(self.support) - 1)
        return tuple(int(self.support[index]) for index in indices)


def normal_mixture_prediction(
    mu: float,
    tau2: float,
    row_totals: np.ndarray,
    column_totals: np.ndarray,
    *,
    tolerance: float = 1e-10,
    quadrature_orders: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048, 4096),
) -> ConditionalMixturePrediction:
    """Integrate Fisher's exact conditional law over N(mu, tau2).

    Parameters describe transported recipient log odds. Source parameter
    uncertainty is not included. Integration must converge in total variation
    and in expected upper-left count divided by the table total.
    """
    mu, tau2 = float(mu), float(tau2)
    if not np.isfinite(mu) or not np.isfinite(tau2) or tau2 < 0:
        raise ValueError("mu must be finite and tau2 finite and nonnegative")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    rows, columns = _integer_binary_margins(row_totals, column_totals)
    total = int(rows.sum())
    lower = max(0, int(rows[0] + columns[0] - total))
    upper = min(int(rows[0]), int(columns[0]))
    support = np.arange(lower, upper + 1)
    log_weights = _log_choose(int(columns[0]), support) + _log_choose(
        int(columns[1]), rows[0] - support
    )

    def result(log_probability, order, error):
        probability = np.exp(log_probability)
        mean = float(probability @ support)
        table = np.array([
            [mean, rows[0] - mean],
            [columns[0] - mean, total - rows[0] - columns[0] + mean],
        ])
        return ConditionalMixturePrediction(
            table, support, probability, log_probability, order, error
        )

    if tau2 == 0 or lower == upper:
        logits = log_weights + mu * (support - lower)
        return result(logits - logsumexp(logits), 0, 0.0)
    if (
        len(quadrature_orders) < 2
        or any(not isinstance(order, int) or order < 2 for order in quadrature_orders)
        or any(b <= a for a, b in zip(quadrature_orders, quadrature_orders[1:]))
    ):
        raise ValueError("quadrature_orders must contain increasing integer orders")

    previous = None
    for order in quadrature_orders:
        nodes, weights = roots_hermitenorm(order)
        positive = weights > 0
        nodes, weights = nodes[positive], weights[positive]
        theta = mu + np.sqrt(tau2) * nodes
        logits = log_weights[None, :] + theta[:, None] * (support - lower)
        conditional_log_p = logits - logsumexp(logits, axis=1, keepdims=True)
        log_probability = logsumexp(
            conditional_log_p + np.log(weights / weights.sum())[:, None], axis=0
        )
        log_probability -= logsumexp(log_probability)
        probability = np.exp(log_probability)
        if previous is not None:
            error = max(
                float(np.abs(probability - previous).sum() / 2),
                float(abs((probability - previous) @ support) / max(total, 1)),
            )
            if np.isfinite(error) and error <= tolerance:
                return result(log_probability, order, error)
        previous = probability
    raise CouplingEstimationRefusal(
        f"normal-mixture quadrature did not converge at order {order} (error {error:g})"
    )
