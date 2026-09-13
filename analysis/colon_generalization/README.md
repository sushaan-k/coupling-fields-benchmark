# Independent blood-to-colon test

An unchanged interaction learned in twelve blood-reference donors was tested
in eleven independent colon-biopsy donors. The protocol and source model were
[published before count access](https://github.com/sushaan-k/coupling-fields-benchmark/commit/b6e6cab01db19baae7ff6aa44ffafcb1330c46a6).
No recipient tuning or replacement cohort was used.

Both assay-pattern distributions reduced mean held-cell deviance by 15.65%
against individual frequencies, 5.42% against protein patterns alone, and
31.13% against independence. The respective multiplicity-adjusted intervals
were 10.89--22.40%, 2.56--9.49%, and 24.85--38.43%. The frozen success criterion
passed. Donor wins were 11/11, 10/11, and 11/11. These are paired donor-bootstrap
intervals conditional on the fixed source fit, not treatment-effect estimates.

`PLAN.md`, `selection.json`, and `freeze.json` retain the pre-access contract.
`results/` contains all five arms, all eleven donor losses, and all 81 query
losses, including unfavorable individual comparisons. There are no raw
cell-level counts or barcodes in this release.

## Reproduce the result

Keep this folder next to `assay_resolution`, which contains the unchanged
reconstruction implementation and source fit. From this folder, run:

```bash
python3 -m pip install numpy scipy threadpoolctl
python3 -m unittest -v test_protocol.py
python3 verify_results.py
python3 report.py
```

`verify_results.py` reconstructs all 55 donor-by-arm predictions from the
released marginal distributions, recomputes losses and the three bootstrap
comparisons, and checks the frozen success criterion. An additional check
using the original research solver independently reproduced all predictions
to within 6.33e-11 and recounted all 891 scoring tables; its numerical report
is `results/verification.json`.

## Reacquire from the source

The [Mennillo et al. study](https://doi.org/10.1038/s41467-024-45665-6) deposits
the public count matrix in [Figshare article 21919356, version 3](https://doi.org/10.6084/m9.figshare.21919356.v3),
under CC BY 4.0. File 54027674 is approximately 3.8 GB. Acquisition verifies
its upstream size and MD5, feature axes, selected donor membership, and counts.
Use an empty work directory for a new run:

```bash
python3 -m pip install h5py
python3 acquire.py --work /path/to/colon-run --public-commit b6e6cab01db19baae7ff6aa44ffafcb1330c46a6
python3 run.py predict --work /path/to/colon-run
python3 run.py evaluate --work /path/to/colon-run
```

`results/first_count_access.json` records the first count access at 22:10:16 UTC,
after publication of the freeze at 22:09:07 UTC. A host failure
interrupted that download before extraction; the unchanged acquisition code
was rerun locally. `results/count_access.json` records the successful retry,
not the earlier attempt. `results/acquisition.json` binds the verified source
file and the separately retained adaptation and scoring arrays.
