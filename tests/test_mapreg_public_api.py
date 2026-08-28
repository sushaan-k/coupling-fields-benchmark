import mapreg

from mapreg import (
    ConditionalAssociationEstimate,
    CouplingFieldRefusal,
    FactorialCouplingDiagnostics,
    FactorialCouplingFit,
    FactorialCouplingRefusal,
    StructuredCouplingFit,
    TablePredictionRefusal,
    association_field,
    conditional_association_coordinates,
    factorial_association_contrast,
    field_coordinates_to_table,
    fit_factorial_coupling,
    fit_structured_coupling_fields,
    ipf_to_margins,
    inverse_permutation_variance_weights,
    multinomial_deviance_per_observation,
    residual_coordinates_to_table,
)


def test_public_api_is_the_declared_coupling_interface():
    assert mapreg.__all__ == [
        "ConditionalAssociationEstimate",
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
        "fit_structured_coupling_fields",
        "field_coordinates_to_table",
        "ipf_to_margins",
        "inverse_permutation_variance_weights",
        "multinomial_deviance_per_observation",
        "residual_coordinates_to_table",
    ]
    assert callable(association_field)
    assert callable(conditional_association_coordinates)
    assert callable(factorial_association_contrast)
    assert callable(fit_factorial_coupling)
    assert callable(fit_structured_coupling_fields)
    assert callable(field_coordinates_to_table)
    assert callable(ipf_to_margins)
    assert callable(inverse_permutation_variance_weights)
    assert callable(multinomial_deviance_per_observation)
    assert callable(residual_coordinates_to_table)
    assert issubclass(CouplingFieldRefusal, ValueError)
    assert issubclass(FactorialCouplingRefusal, ValueError)
    assert issubclass(TablePredictionRefusal, ValueError)
    assert ConditionalAssociationEstimate.__module__ == "mapreg.coupling_fields"
    assert StructuredCouplingFit.__module__ == "mapreg.coupling_fields"
    assert FactorialCouplingDiagnostics.__module__ == "mapreg.factorial_coupling"
    assert FactorialCouplingFit.__module__ == "mapreg.factorial_coupling"
