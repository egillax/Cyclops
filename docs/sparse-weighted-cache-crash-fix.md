# Sparse logistic weighted-cache crash

## Cause and fix

The September 2026 synthetic crash report identifies an uninitialized weighted
column cache in `computeSparseLogisticGradientAndHessian`. Fitted probabilities
can be ready while `sparseLogisticWeightedColumnViews` is empty. Accessing its
first sparse column is undefined behavior and crashed on Linux and Windows.
Dense/intercept branches can instead silently scan an empty weighted-row list.

The gradient/Hessian entry point now calls
`ensureSparseLogisticWeightedColumnViews()` whenever `useWeights` is true,
independently of fitted-state readiness. This method returns immediately for an
already valid cache. `setWeights()` already synchronizes state and invalidates
the cache; the next weighted access rebuilds it for the new fold. Other weighted
accessors in ModelSpecifics.hpp already guard their access with the same ensure
method. No convergence, prior, intercept, or auto-search settings were changed.

Derivative comparisons exposed a second stale-state problem: public gradient
and Hessian-diagonal queries used legacy state after fitted-response updates.
Both queries now synchronize sparse logistic state before computing derivatives.

## Validation

Built from the existing dirty Cyclops-perf-ideas worktree into the isolated
library `/tmp/cyclops-cache-fix-library`. The user's installed package was not
replaced. Existing unrelated edits were preserved.

The original `/tmp/fl-cyclops-crash-investigation/reproduce.R` completed on Linux
with the reported settings: 57,406 rows, 63 columns, 10-fold auto-CV, one thread,
seed 47, tolerance 2e-6, unpenalized intercept, initial variance 0.01.

Against legacy from the same rebuilt package:

| Check | Result |
| --- | --- |
| Return status | SUCCESS for both |
| Selected variance | 0.01796606 for both |
| Maximum absolute coefficient difference | 2.047251e-13 |
| Absolute log-likelihood difference | 4.547474e-12 |
| Printed auto-CV search score traces | Identical at printed precision |

`tests/testthat/test-weighted-cache.R` runs a child R process to isolate native
crashes. Its fixture checks sparse intercepts and continuous columns, explicit
indicator and dense storage, repeated zero/unit/fractional weights, warm starts,
coefficients, gradients, Hessian diagonals, held-out likelihoods, and auto-CV
with one and three threads against legacy. It passes on Linux. Windows execution
and production confirmation remain outstanding.

Existing gradient and weighting tests pass. Existing CV tests pass with
NOT_CRAN=true (25 assertions; three explicitly disabled tests remain skipped).
`git diff --check` passes.

Logs and full-reproducer coefficient results are in
`/tmp/cyclops-cache-fix-{auto,legacy}-final.log` and
`/tmp/cyclops-cache-fix-{auto,legacy}-result.rds`.

## Rebuild for deployment

This is native code: pulling source or restarting R alone cannot fix an already
installed binary. Rebuild and install the modified worktree, then restart every
R process using Cyclops. Confirm the loaded package/library path before rerunning
the original reproducer and the production workload.
