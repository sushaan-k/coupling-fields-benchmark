# GSE309593 public authorization and tag checklist

No item below has been executed by this pre-outcome freeze. Do not access a GSE309593 H5,
ADT CSV, identifier, barcode, or assay value until items 1--4 are complete.

1. Review the source-only amendment, protocol JSON, runner, tests, templates,
   and this checklist. Confirm the historical candidate file is unchanged.
2. Run the bound unit and artifact-integrity tests. Create an annotated
   `gse309593-independent-study-v1-protocol` tag and push it. Verify its tag
   object, peeled commit, bound runtime specification, runner, tests, protocol,
   complete supersession matrix, and sequential-candidate disclosure from a
   fresh HTTPS fetch. Record a scan confirming that no GSE309593 assay,
   identifier, barcode, bridge, private-state, prediction, or score artifact is
   present.
3. Close and publish the GSE288020 development artifact under its annotated
   development tag. This requires only the corrected salted 7/7 MGUS split.
   The separate nine-donor MM diagnostic may pass or fail; it is not a branch
   gate here.
4. Fit every fixed method on the 14 MGUS development donors and publish the
   GSE288020 development artifact containing the nested source model. Populate
   `source_authorization_v1.json` from the disabled template. Bind the public
   canonical GSE288020 protocol and development annotated tags and commits; the
   exact STARTED/FINISHED development ledger; its runtime specification,
   artifact path, bytes, status, protocol commit, and hash; the canonical
   nested-model hash; protocol, amendment, candidate, runner, and transitive
   hashes; zero MM outcome use; and zero GSE309593 access. Require all 14 MGUS
   donors, all five methods with `VALID` status, empty method refusals,
   `classical_head_to_head_ready=true`, and `external_study_ready=true`. Set
   status to
   `SOURCE_MODEL_AND_RECIPIENT_RNA_ACCESS_AUTHORIZED`, set
   `recipient_rna_access_authorized` to true, publish
   `gse309593-independent-study-v1-source-authorized`, and verify it from a
   fresh fetch.
5. Run `claim rna`. Publish the one-line attempt ledger under
   `gse309593-independent-study-v1-rna-attempt` and verify it. Only then run
   `run-rna` with scratch storage and two outside-repository private paths.
   Confirm source-equivalent RNA-only QC precedes salted selection and that all
   sparse matrix invariants pass. Publish the exact three-record ledger and
   semantically validated RNA artifact under
   `gse309593-independent-study-v1-rna` and verify it.
6. Run `claim adt`, publish and verify
   `gse309593-independent-study-v1-adt-attempt`, then run `run-adt` with the
   frozen identifier bridge and an outside-repository ADT-state path. Confirm
   every selected identifier occurs exactly once and every subject-marker has a
   marginal tie diagnostic and support mask; unsupported markers must carry a
   512/0 sentinel margin and remain on the global feature axis. Publish and
   semantically verify `gse309593-independent-study-v1-adt`.
7. Run `claim prediction`, publish and verify
   `gse309593-independent-study-v1-prediction-attempt`, then run
   `run-prediction`. Confirm the output reports zero joint recipient tables and
   zero simultaneous private-state access, excludes unsupported subject-marker
   pairs through frozen margins, retains at least 18 subjects and all seven
   batches, and contains all five predictions. Publish and independently verify
   `gse309593-independent-study-v1-predictions`.
8. Populate `score_authorization_v1.json` from its disabled template. Bind the
   prediction tag, commit, path, hash, byte count, RNA and ADT stage hashes,
   source authorization, protocol, runner, and all transitive hashes. Set
   status to `JOINT_SCORING_AUTHORIZED`, attest that zero recipient joint tables
   were formed before authorization, and set `outcome_access_authorized` to true.
   Publish and verify `gse309593-independent-study-v1-score-authorized`.
9. Run `claim score`, publish and verify
   `gse309593-independent-study-v1-score-attempt`, then run `run-score` once
   with the two private state artifacts. Publish the completed negative,
   confirmation pass, or terminal refusal under
   `gse309593-independent-study-v1-result`, then run `verify-result` against the
   fresh public tag for a completed score. Never amend a criterion or rerun a
   claimed stage.

Fresh verification must record the annotated tag object, peeled commit, exact
file tree, runtime, attempt-ledger semantics, prerequisite chain, payload
semantics, status, byte count, and SHA-256 of every stage artifact. A GitHub
release page is not a substitute for verifying the Git tag itself.
