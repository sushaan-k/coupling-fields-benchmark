# Coupling-field exactness and the classical boundary

## Setup

Let `P` be a strictly positive `r x c` joint probability table. Let `H_r`
and `H_c` be orthonormal contrast matrices whose columns span the vectors
orthogonal to the all-ones vector. Define the interaction coordinates

`eta(P) = H_r^T log(P) H_c`.

The implementation calls the lifted, zero-row/column-sum version of these
coordinates the coupling field. Row and column margins are supplied separately.

## Exact population statement

**Proposition.** If

`Q_ij = a_i P_ij b_j / Z`

for positive row tilts `a`, column tilts `b`, and normalizer `Z`, then
`eta(Q) = eta(P)`. Conversely, two strictly positive tables have the same
interaction coordinates if and only if they differ by such a separable tilt.
For every pair of strictly positive target margins with the same total, there
is a unique table with those margins and interaction coordinates `eta(P)`.
Iterative proportional fitting reconstructs that table.

**Proof.** The log of a separable tilt adds one row term, one column term, and a
constant to `log(P)`. Left and right multiplication by the centered contrast
bases removes all three, proving invariance. Conversely, if the interaction
coordinates agree, `log(Q) - log(P)` is orthogonal to the row-by-column
interaction subspace. It therefore has the additive form `alpha_i + beta_j`,
which exponentiates to a separable tilt. A strictly positive seed has a unique
diagonal scaling to any strictly positive compatible margins, up to the
irrelevant reciprocal scaling of the two diagonal factors. This is the table
returned by iterative proportional fitting. QED.

The result is exact for population tables. Sampling error, zero cells,
pseudocounts, state discretization, and an interaction that changes between
source and target are outside its premise.

## Why independence residuals do not share the guarantee

For total `N`, row probabilities `p_i`, and column probabilities `q_j`, the
Pearson residual is

`R_ij = sqrt(N) (P_ij - p_i q_j) / sqrt(p_i q_j)`.

It depends explicitly on the margins and on depth. A fixed interaction does not
fix `R`. In a `2 x 2` table with odds ratio `omega`, row-one margin `r`, and
column-one margin `c`, the `(1,1)` probability `t` solves

`t(1-r-c+t) / ((r-t)(c-t)) = omega`.

The Pearson association is

`phi = (t-rc) / sqrt(r(1-r)c(1-c))`,

which varies with `r` and `c` at fixed `omega`. With `omega=9`, balanced margins
give `phi=0.500`; margins `r=0.8`, `c=0.3` give `phi=0.263`. Signed Poisson
deviance residuals have the same dependence through the independence mean.

The deterministic `3 x 3` sweep in
`results/coupling_margin_invariance_simulation.json` gives every method the
exact target margins and transfers the complete source residual matrix, not a
lower-dimensional approximation. Field reconstruction remains exact to
floating-point precision. At the terminal margin shift, multinomial deviance
per observation is `0.006229` for Pearson residual transfer and `0.004839` for
signed-deviance residual transfer, compared with less than `1e-12` for the
field.

## Classical boundary

A saturated Poisson log-linear model can be written as

`log(mu_ij) = lambda + rho_i + kappa_j + gamma_ij`,

with centered constraints on `gamma`. Under those constraints, `gamma` is the
coupling field. Corner-coded log odds ratios contain the same information by an
invertible change of basis. A classical model that estimates and transfers this
interaction term, then refits row and column effects to the target margins,
ties the coupling-field reconstruction exactly at the population level.

The defensible comparison is therefore precise: coupling fields add a
margin-invariant interaction parameterization over direct transfer of Pearson
or signed-deviance residuals from an independence model. They do not constitute
a new estimand relative to saturated log-linear interaction modeling. Any
further contribution must come from the conditional null correction, structured
cross-entity estimator, held-unit refusal rule, or experimental design, and must
be validated separately.
