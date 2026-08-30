"""Public API for conditional coupling fields in paired assays."""

from .coupling_fields import (
    ConditionalAssociationEstimate,
    CouplingFieldRefusal,
    StructuredCouplingFit,
    association_field,
    conditional_association_coordinates,
    factorial_association_contrast,
    fit_structured_coupling_fields,
    inverse_permutation_variance_weights,
)
from .context_conditional_coupling import (
    ContextConditionalCouplingFit,
    fit_context_conditional_log_odds,
    predict_context_log_odds,
)
from .factorial_coupling import (
    FactorialCouplingDiagnostics,
    FactorialCouplingFit,
    FactorialCouplingRefusal,
    fit_factorial_coupling,
)
from .table_prediction import (
    TablePredictionRefusal,
    field_coordinates_to_table,
    ipf_to_margins,
    multinomial_deviance_per_observation,
    residual_coordinates_to_table,
)


__all__ = [
    "ConditionalAssociationEstimate",
    "ContextConditionalCouplingFit",
    "CouplingFieldRefusal",
    "FactorialCouplingDiagnostics",
    "FactorialCouplingFit",
    "FactorialCouplingRefusal",
    "StructuredCouplingFit",
    "TablePredictionRefusal",
    "association_field",
    "conditional_association_coordinates",
    "factorial_association_contrast",
    "fit_factorial_coupling",
    "fit_context_conditional_log_odds",
    "fit_structured_coupling_fields",
    "field_coordinates_to_table",
    "ipf_to_margins",
    "inverse_permutation_variance_weights",
    "multinomial_deviance_per_observation",
    "predict_context_log_odds",
    "residual_coordinates_to_table",
]
