# GSE158769 terminal source-schema refusal

The frozen v1.1 development attempt stopped before table construction, model
fitting, or pilot evaluation. The deposited antibody rows use labels such as
`CD3_protein`, whereas the frozen source whitelist contained supplement-style
labels such as `CD3`. The runner therefore could not resolve the first ADT
marker and raised `KeyError: ('adt', 0)`.

The failed stream selected only the 154 calibration and pilot donors (78,848
cells). It tokenized the nine frozen RNA rows. Four gene rows also satisfied
provisional same-name ADT aliases in memory, but no deposited `_protein` row
was tokenized. No held cell column or held numeric value was selected. No state,
2-by-2 table, fitted model, pilot statistic, or result file was produced by the
runner.

A post-failure scan read only the first field of each feature row and established
the exact technical suffix. It parsed no numeric value. Because the protocol
explicitly makes a missing frozen feature a terminal schema refusal, and the
failed development stream had already parsed development RNA values, the
source adapter was not amended and development was not rerun. The complete
machine-readable access record is
`results/development/gse158769_development_v1.json`.

This outcome is a source-schema refusal, not a negative biological pilot. It
does not supply confirmation evidence and cannot be used to tune a subsequent
candidate.
