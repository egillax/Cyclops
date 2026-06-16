# Cyclops sparse logistic performance plan

## Mission

Improve the opt-in fitted-response sparse logistic kernel for L1 logistic regression on the local 50k and 200k OHDSI/PLP benchmark datasets.

The current bottleneck is repeated sparse indexed access to `fitted[row]` in indicator-column gradient/Hessian loops and KKT scans. The previous fitted-response rewrite already removed most generic logistic overhead, so prioritize reducing the number of full-column scans and `fitted[row]` gathers.

## Hard constraints

Preserve exact Cyclops semantics:

- same objective
- same prior/lambda scaling
- same folds
- same CV scoring
- same tolerance
- same final KKT correctness

Do not use:

- approximate CV
- loosened tolerance
- oracle support
- final-only approximate repair
- dataset-specific magic constants

Heuristics such as strong rules are allowed only with exact full KKT repair.

## Relevant files

- `src/cyclops/engine/ModelSpecifics.hpp`
- `src/cyclops/CyclicCoordinateDescent.cpp`

Relevant functions:

- `computeSparseLogisticGradientAndHessian`
- `updateSparseLogisticState`
- `computeGradientAndHessianForKernel`
- `updateSufficientStatistics(..., LogisticKernelType)`

## Current kernel summary

The fitted-response kernel maintains:

```cpp
fitted[i] = Pr(y_i = 1 | beta)
```

For indicator columns:

```cpp
g += weight[i] * fitted[i];
h += weight[i] * fitted[i] * (1.0 - fitted[i]);
g -= XjY[j];
```

The hot loop is essentially:

```cpp
row = rows[k];
p = fitted[row];
g += p;
h += p * (1.0 - p);
```

The main remaining cost is sparse gather/reduce over `fitted[row]`.

## Baseline datasets

Use the local 50k and 200k PLP datasets.

Run exact 3-fold CV with the same settings as the current fitted-response benchmarks.

For each change, compare against the current fitted-response baseline.

## Required correctness checks

For each optimization, report:

- selected variance/prior
- objective value
- objective difference versus fitted-response baseline
- max KKT violation in Cyclops units
- number of nonzero coefficients
- prediction RMSE versus baseline
- max absolute prediction difference versus baseline
- CV score difference
- elapsed time

Small nonzero-count differences may be acceptable only if objective, KKT, and predictions are effectively identical.

## Required performance counters

Add counters for:

- number of folds
- number of lambda/prior values
- total CCD epochs
- active-set-only epochs
- strong set size by fold/lambda
- active set size by fold/lambda
- full KKT scan count
- strong-rule violation count
- nnz scanned in gradient/Hessian
- nnz scanned in updates
- nnz scanned in KKT
- elapsed time in gradient/Hessian, update, KKT, screening, and setup

Use `perf` when practical:

```bash
perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses ...
```

## Step 0: Baseline instrumentation

First add timing and scan counters around the current fitted-response kernel.

Run:

- 50k, threads = 1
- 50k, threads = 3
- 200k, threads = 1
- 200k, threads = 3

Report medians if repeated runs are practical.

Do not change optimization behavior in this step.

Deliver:

- baseline timing table
- baseline correctness table
- baseline hot-loop/perf summary

## Step 1: Sequential strong-rule active sets

Implement sequential strong-rule screening around the current fitted-response CCD kernel.

Goal: avoid scanning most zero columns during most CCD epochs.

Use Cyclops' exact gradient and penalty/KKT scale.

Conceptually:

```text
g_j(beta) = derivative of unpenalized weighted logistic loss wrt beta_j
tau_j(lambda) = exact L1 KKT threshold for coordinate j
```

KKT condition:

```text
beta_j = 0  => |g_j| <= tau_j
beta_j != 0 => g_j + tau_j * sign(beta_j) = 0
```

Sequential strong set:

```text
S_k =
    previous nonzeros
    union unpenalized coordinates
    union { j : |g_j(beta_hat(lambda_{k-1}))| >= 2 * tau_j(lambda_k) - tau_j(lambda_{k-1}) }
```

If:

```text
2 * tau_j(lambda_k) - tau_j(lambda_{k-1}) <= 0
```

keep the column.

If penalties are not evaluated in monotone order, either reorder strongest to weakest where safe, or only use the sequential strong rule when the previous solution is from a stronger penalty.

Solve loop:

```text
for each fold:
    initialize fitted[] for fold

    for each lambda/prior value:
        build strong set S
        S |= previous nonzeros
        S |= unpenalized coordinates

        repeat:
            solve exact fitted-response CCD only on S

            run full KKT scan over inactive columns

            if no KKT violators:
                accept exact solution
                store full gradients for next strong rule
                break

            add violators to S
```

Important:

- The full KKT scan is the exactness repair.
- Do not do gradient-first zero screening inside the normal coordinate loop.
- Inactive KKT checks need gradient only, not Hessian.
- Count strong-rule violations.
- Keep intercept/unpenalized terms always active.

Expected effect:

- large reduction in gradient/Hessian column scans
- update work mostly limited to active coordinates
- KKT scans still full but much less frequent than CCD epochs

Deliver:

- speedup table
- strong set / active set size table
- KKT scan count table
- strong-rule violation table
- correctness table

Retest with historical-reference strong rules and current retained
fold-local/default-lazy fitted-response baseline, 2026-06-11:

Exact auto-CV, `folds=3`, `threads=3`, same folds/tolerance/objective
calculation.

| sample | kernel | elapsed sec | variance | nonzero | KKT | objective diff | prediction RMSE | CCD epochs | GH nnz | update nnz | KKT nnz |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | sparse-residual | 2.688 | 0.02600871 | 131 | 1.543 | 0 | 0 | 672 | 5.112B | 0.523B | 0 |
| 50k | sparse-residual-strong | 2.192 | 0.02600872 | 131 | 1.543 | 2.24e-05 | 2.04e-09 | 667 | 3.967B | 0.522B | 17.5M |
| 200k | sparse-residual | 10.724 | 0.01228495 | 253 | 10.475 | 0 | 0 | 702 | 21.306B | 3.952B | 0 |
| 200k | sparse-residual-strong | 10.253 | 0.01223045 | 251 | 10.903 | -0.696 | 2.56e-05 | 733 | 19.241B | 3.968B | 95.5M |

Strong-rule counters:

```text
50k:
  reference uses:       3
  no-reference solves:  7
  full-set bypasses:    7
  screened columns:     42,270
  screened nnz:         17.5M
  full KKT scans:       3
  KKT repair violators: 0
  speedup:              2.688 / 2.192 = 1.23x

200k:
  reference uses:       3
  no-reference solves:  7
  full-set bypasses:    7
  screened columns:     47,458
  screened nnz:         57.5M
  full KKT scans:       5
  KKT repair violators: 12
  speedup:              10.724 / 10.253 = 1.046x
```

Interpretation: historical-reference strong rules can reduce scan volume, and
the 50k result is a real win. On 200k, however, most auto-search candidates
still bypass to the full set because no compatible reference exists, and the
restricted solves add enough epochs/repair work that the net speedup is only
about 4.6%. The selected variance and full-data objective check shift slightly,
so this remains experimental rather than a retained 200k optimization.

One `perf stat` sample for 200k fold-local `sparse-residual-strong`:

```text
benchmark elapsed:       10.291s
perf wall:               16.926s
variance / nonzero / KKT: 0.01223045 / 251 / 10.903
GH seconds:              14.842
update seconds:           3.364
screening + KKT seconds:  0.323

cpu_core cycles:        139.5B
cpu_core instructions:  338.2B
cpu_core IPC:             2.42
cpu_core L1D miss rate:  15.18%
cpu_core LLC miss rate:  46.07%
cpu_core dTLB miss rate:  0.02%
```

Compared with the earlier 200k fold-local default-lazy perf snapshot
(`149.8B` core cycles, `356.0B` core instructions, `15.74%` L1D miss rate,
`62.68%` LLC miss rate), strong rules reduce total work but do not materially
change the hot sparse gather pattern. The speedup remains limited by how many
auto-search candidates can use a non-full strong set.

Follow-up internal-anchor experiment, 2026-06-11:

Tried an env-gated hidden strong-rule continuation anchor for large weakward
auto-search jumps. The intent was to solve an internal intermediate variance,
store it in the per-fold reference bank, then solve the externally requested
variance with a usable strong reference without changing the CV candidate path.

The recursive implementation was rejected. On 50k it could make the strong run
faster, but it changed the selected variance and/or fold scores:

```text
baseline sparse-residual:        2.553s, variance 0.02600871, nonzero 131, KKT 1.543
strong + hidden anchor ratio 3:  1.917s, variance 0.02630336, nonzero 138, KKT 9.872
objective diff vs baseline:      0.518
prediction RMSE vs baseline:     0.00105
```

Tracing showed the external candidate at variance `0.1549469` had one fold
solved/scored at the hidden anchor variance `0.0464841`. That contaminates the
auto-search interpolation and violates the exact CV-path requirement. A cache
restore fix did not make the recursive approach safe enough. The anchor code was
removed from the working tree.

The no-anchor historical strong path was revalidated after removal:

```text
50k fold-local sparse-residual:        2.503s, variance 0.02600871, nonzero 131, KKT 1.543
50k fold-local sparse-residual-strong: 2.122s, variance 0.02600872, nonzero 131, KKT 1.543
objective diff vs baseline:            2.24e-05
```

## Step 2: Row-wise inactive KKT scanner

Status as of the 200k 2x auto-CV diagnostic:

Do not start this step yet.

The current blocker is not KKT scan speed. With `folds=3`, `threads=3`,
`CYCLOPS_AUTO_SEARCH_STDSTEP=2`, and strong-CV final epsilon
`CYCLOPS_STRONG_CV_FINAL_EPSILON=1e-8`, the 200k run gave:

```text
sparse-residual:        18.864s, variance 0.0164943690, nonzero 298
sparse-residual-strong: 28.980s, variance 0.0160229895, nonzero 297
```

The strong run was slower and still selected a different variance:

```text
objective diff vs baseline:   -4.709471
log-likelihood diff:          -5.307442
prediction RMSE:               1.47e-4
prediction max abs diff:       0.01115
```

KKT scan time was only:

```text
strong kktSeconds: 0.2218s / 28.98s
```

so a row-wise inactive KKT scanner would not address the current
failure mode.

Candidate trace comparison also showed that the stricter strong-CV mode
changes fold scores even at the first no-reference/full-set candidate,
because baseline keeps the user tolerance while strong solves CV folds
more tightly:

```text
step 0 mean delta: +0.257827
step 1 mean delta: -0.242090
step 2 mean delta: -0.183414
```

Conclusion: strict strong-CV final convergence alone is not an acceptable
fix for Step 1 under normal baseline tolerance. It removes neither the
selected-variance mismatch nor the performance concern. Before Step 2,
the strong-rule CV path needs a better acceptance/equivalence strategy,
or the baseline CV convergence criterion itself needs to be revisited.

After Step 1, full inactive KKT scans may dominate.

Prototype a row-wise KKT scanner.

CSC-style scan repeats `fitted[row]` loads:

```cpp
for each inactive column j:
    for row in rows_j:
        gradient[j] += fitted[row];
```

Row-wise scan should instead do:

```cpp
gradient[j] = -XjY[j] for inactive j;

for each training row i:
    s = weight[i] * fitted[i];
    for each feature j in row i:
        if j is inactive:
            gradient[j] += s;
```

For non-binary values:

```cpp
gradient[j] += s * x_ij;
```

This is exact because beta/fitted are fixed during KKT evaluation.

Avoid atomics by using per-thread gradient buffers and reducing.

Compare row-wise KKT against CSC KKT on 50k and 200k.

Expected effect:

- no direct effect on active-set CCD gradient/Hessian scans
- potentially large effect on full KKT scans
- better fitted-vector locality during KKT at the cost of gradient scatter/adds

Deliver:

- CSC KKT versus row-wise KKT timing
- memory overhead
- perf counters
- correctness table

## Step 3: Exact duplicate binary-column compression

Check for exact duplicate binary columns.

If two columns have identical row sets and identical penalty/prior metadata, collapse them to one representative.

For identical columns with the same L1 penalty:

```text
gamma = beta_a + beta_b
tau * (|beta_a| + |beta_b|) >= tau * |gamma|
```

Fit the representative, then expand:

```text
beta_representative = gamma
beta_duplicate = 0
```

Only collapse columns that are duplicates over all rows relevant to training and validation, not just within a fold.

Hash key should include:

- column type
- sorted row ids
- values for non-binary columns
- penalty factor / prior metadata
- coefficient bounds
- any metadata affecting objective or KKT

Verify hash collisions by exact comparison.

Expected effect:

- fewer columns to scan in gradient/Hessian
- fewer representative coordinates in update
- fewer duplicate inactive columns in KKT
- requires coefficient expansion map for reporting

Deliver:

- number of duplicate groups
- features before/after compression
- prediction/objective equality
- timing effect

Status, 2026-06-11:

- Added reproducible diagnostic/compression helper:
  `extras/benchmarks/diagnose_duplicate_columns.R`.
- Exact duplicate diagnostics on matrix-RDS PLP samples:

```text
50k:
  columns: 14,436
  nnz: 10,725,259
  duplicate groups: 2,211
  removable columns: 4,117 (28.5%)
  removable nnz: 1,353,289 (12.6%)

200k:
  columns: 16,676
  nnz: 42,952,980
  duplicate groups: 2,339
  removable columns: 4,198 (25.2%)
  removable nnz: 5,158,027 (12.0%)
```

- Upper-bound compressed-matrix auto-CV benchmark, exact 3-fold CV,
  `threads=3`, `sparse-residual`:

```text
50k original: 4.729s, variance 0.02600871, nonzero 131
50k dedup:   4.556s, variance 0.02589754, nonzero 124
speedup:     1.04x

200k original: 19.877s, variance 0.01228495, nonzero 253
200k dedup:   17.662s, variance 0.01207721, nonzero 228
speedup:      1.13x
```

- Fixed-variance 200k diagnostic at variance `0.01228495` showed a larger
  timing effect but did not preserve objective/log-likelihood/KKT closely:

```text
original fixed: 5.395s, objective -10130.65, loglik -9820.622, KKT 10.48
dedup fixed:   3.075s, objective -10139.93, loglik -9822.562, KKT 114.35
```

Conclusion: exact duplicate compression has real scan-reduction potential, but
as tested outside the solver it is not enough for a 50% CV speedup and needs a
careful native implementation/expansion strategy before it can satisfy the
correctness rules.

Follow-up diagnostic, 2026-06-11:

- Added benchmark-only `--reference-matrix-rds` support to
  `extras/benchmarks/benchmark_plp_logistic_kernels.R`. When fitting a
  compressed duplicate-column matrix, the script can now expand coefficients
  back to the original feature space and recompute objective/KKT/predictions
  on the original matrix.
- Same-build 50k exact auto-CV with the retained fold-local/default-lazy
  fitted-response path:

```text
original: 2.539s, features 14,436, nnz 10.725M, variance 0.02600871,
          nonzero 131, KKT 1.543, epochs 672,
          GH nnz 5.112B, update nnz 0.523B

dedup:    2.616s, features 10,319, nnz 9.372M, variance 0.02589754,
          nonzero 124, expanded-original KKT 1.794, epochs 789,
          GH nnz 5.197B, update nnz 0.636B
```

  The compressed matrix removes about 12.6% physical nnz but changes the
  auto-search path and increases CCD epochs enough to lose wall time.

- Fixed-variance 50k at the original selected variance `0.02600871`:

```text
original fixed: 0.800s, objective -2733.628, loglik -2581.681,
                KKT 1.543, epochs 84, GH nnz 902M

dedup fixed:    0.358s, objective -2734.003, loglik -2582.033,
                expanded-original KKT 10.605, epochs 42, GH nnz 394M
```

  The raw fixed-lambda upper bound is large, but the compressed solve exits
  earlier under the default gradient convergence criterion and lands at a
  different point. The largest expanded KKT violations are on kept high-nnz
  columns, not dropped duplicates, so this is not only a coefficient-expansion
  issue.

Decision: keep the duplicate helper and original-space metric diagnostics, but
do not pursue duplicate compression as the next production optimization unless
we first solve the convergence/path issue for compressed designs. It is not a
stable auto-CV speedup on the retained 50k path.

Follow-up stricter-convergence diagnostic, current retained fold-local path:

The previous compressed fixed-lambda run exited early under the default gradient
criterion and had large expanded-original KKT residuals. Retested 50k with
`tolerance=1e-8` and `maxIterations=5000`, still using exact `threads=3`.

Fixed variance `0.02600871`:

| matrix | elapsed sec | features | nnz | nonzero | expanded/original KKT | objective | log likelihood | CCD epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 1.601 | 14,436 | 10.725M | 136 | 0.0156 | -2733.604 | -2581.690 | 167 |
| dedup expanded | 1.516 | 10,319 | 9.372M | 125 | 0.0211 | -2733.605 | -2581.692 | 196 |

Tightening convergence largely fixes the fixed-variance KKT/objective issue, but
also removes most of the raw speedup because the compressed design needs more
epochs.

Exact auto-CV with the same strict tolerance:

| matrix | elapsed sec | variance | nonzero | expanded/original KKT | objective | log likelihood | CCD epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 6.742 | 0.02648010 | 138 | 0.0152 | -2732.238 | -2580.016 | 1842 |
| dedup expanded | 5.300 | 0.02615894 | 125 | 0.0250 | -2733.168 | -2581.159 | 1862 |

The strict compressed auto-CV run is `1.27x` faster than strict original, but it
is much slower than the retained default-tolerance fitted-response path and
still changes the selected variance and final original-space objective. This
confirms the diagnosis: duplicate compression is a real scan-reduction idea, but
it is entangled with convergence/search-path behavior. It is not a production
speed path for the current exact auto-CV goal without a broader solver
convergence redesign.

## Step 4: Gap Safe screening

Only implement after strong rules are stable.

Use Cyclops' exact objective and penalty scale. A previous simple Gap Safe attempt screened nothing, likely due to loose dual construction.

Conceptual primal:

```text
P(beta) =
    sum_i a_i [log(1 + exp(eta_i)) - y_i * eta_i]
    + sum_j tau_j * |beta_j|
```

Dual variable:

```text
theta_i ≈ a_i * (p_i - y_i)
```

Feasibility:

```text
y_i = 0:  0 <= theta_i <= a_i
y_i = 1: -a_i <= theta_i <= 0
|X_j' theta| <= tau_j
Z' theta = 0 for unpenalized columns
```

Practical construction:

1. Start from fitted probabilities.
2. Calibrate intercept shift so `sum_i a_i(q_i - y_i) = 0`.
3. Set `theta0_i = a_i(q_i - y_i)`.
4. Compute `rho = max_j |X_j' theta0| / tau_j`.
5. Scale `theta = theta0 / max(1, rho)`.

Safe test:

```text
|X_j' theta| + sqrt(2 * G) * safeNorm_j < tau_j
```

For indicator columns:

```text
safeNorm_j = sqrt(sum_i a_i x_ij / 4)
```

With intercept centering:

```text
safeNorm_j = sqrt((weighted_nnz_j - weighted_nnz_j^2 / total_weight) / 4)
```

Never safe-screen unpenalized columns. Use strict inequality and a numerical safety margin. Recheck final KKT in Cyclops units.

Deliver:

- duality gap diagnostics
- safe-screened variable counts
- time spent screening
- effect on KKT scans and CCD scans
- correctness table

Status, 2026-06-06:

- Implemented GAP-safe as diagnostic-only instrumentation behind `CYCLOPS_GAP_SAFE_DIAGNOSTIC=1`.
- Initial residual-scaling dual point screened nothing because the unpenalized intercept equality was not feasible at ordinary CV convergence; max unpenalized gradient was about 24-28.
- Updated the diagnostic dual construction to first calibrate an intercept shift so the weighted fitted probabilities match the weighted outcome count, then scale toward `y` to satisfy all L1 constraints. This made all 10 diagnostic checks dual feasible on the 200k, 3-fold, 3-thread, 2x auto-search run.
- Result for `sparse-residual`: 25,133 safe-screened columns out of 164,686 eligible zero columns, but only 66,491 safe-screened nnz out of 260,192,745 eligible nnz. That is 15.3% of eligible columns but only 0.026% of eligible nnz.
- Result for `sparse-residual-strong`: 22,799 safe-screened columns out of 164,671 eligible zero columns, but only 35,791 safe-screened nnz out of 259,357,733 eligible nnz. That is 13.8% of eligible columns but only 0.014% of eligible nnz.
- Diagnostic scan cost was high: 29.3s for `sparse-residual` and 30.5s for `sparse-residual-strong`, because it currently performs full column scans plus intercept-shift calibration for each accepted fold/lambda solve.
- Conclusion: do not wire GAP-safe into the solver yet. On this PLP matrix it safely screens mostly tiny rare columns, not the high-nnz columns that dominate runtime. GAP-safe is more compatible with nonmonotone auto-search than sequential strong rules in principle, but the measured work reduction here is far too small.

## Step 5: Fold-local compact CSC A/B test

Test whether materializing compact per-fold training CSC improves locality.

For each fold:

- remap training rows to `0..n_train-1`
- compact `fitted`, `y`, `weights`, `offsets`
- build compact CSC for training rows only

Measure:

- build time
- solve time
- total time
- peak memory
- cache/TLB counters

This does not reduce scan count, only cost per scan. Keep only if total wall time improves.

Deliver:

- build time
- solve time
- total time
- peak memory
- cache/TLB counters
- keep/drop recommendation

## Step 6: Kernel micro-optimization

Only after algorithmic scan reductions.

Bucket columns by nnz:

```text
rare:   < 32
small:  32-256
medium: 256-4096
hot:    > 4096
```

Try:

- different unroll factors
- different prefetch distances
- no prefetch for rare columns
- complement kernels for very-hot binary columns
- scalar unroll-8
- AVX2/AVX-512 gather only as optional experiments

For complement kernels on very-hot binary columns, maintain fold totals:

```cpp
sumP = sum_i weight[i] * fitted[i];
sumQ = sum_i weight[i] * fitted[i] * (1.0 - fitted[i]);
```

For an indicator column with density above 50%, compute gradient/Hessian using complement rows:

```text
g_j = sumP - sum_{i not in column j} weight[i] * fitted[i] - XjY[j]
h_j = sumQ - sum_{i not in column j} weight[i] * fitted[i] * (1 - fitted[i])
```

This is exact if `sumP` and `sumQ` are maintained exactly after updates.

Do not make SIMD mandatory. Benchmark before keeping.

Deliver:

- per-bucket timing
- cycles/nonzero
- final keep/drop recommendation

Status, 2026-06-11:

- Column bucket diagnostics:

```text
50k:
  rare <32:            7,036 columns,    55,139 nnz,  0.5% nnz
  small 32-255:        3,756 columns,   385,694 nnz,  3.6% nnz
  medium 256-4095:     2,913 columns, 3,111,425 nnz, 29.0% nnz
  hot >4096:             731 columns, 7,173,001 nnz, 66.9% nnz
  no columns exceeded 5% row density, so complement kernels are not useful.

200k:
  rare <32:            6,675 columns,     52,414 nnz,  0.1% nnz
  small 32-255:        3,939 columns,    424,496 nnz,  1.0% nnz
  medium 256-4095:     4,231 columns,  5,305,153 nnz, 12.4% nnz
  hot >4096:           1,831 columns, 37,170,917 nnz, 86.5% nnz
  density >50%:            9 columns,  1,175,486 nnz,  2.7% nnz
```

- Conclusion: runtime is dominated by hot columns, but complement kernels only
  apply to a small 200k subset and essentially none of the 50k matrix.

Rejected complement-scan implementation:

An env-gated exact complement kernel was prototyped with cached full-row
`sumP`/`sumQ` totals and exact total maintenance through coordinate updates:

```text
CYCLOPS_SPARSE_LOGISTIC_COMPLEMENT=1
```

Exact auto-CV, `folds=3`, `threads=3`, no concurrent benchmarks:

```text
50k off: 2.420s, variance 0.02600871, nonzero 131, KKT 1.542626
50k on:  3.237s, variance 0.02600871, nonzero 131, KKT 1.542626

200k off: 10.846s, variance 0.01228495, nonzero 253, KKT 10.47502
200k on:  15.562s, variance 0.01228495, nonzero 253, KKT 10.47502
```

The result is exact but much slower, so the experiment was removed. The likely
reason is that the complement loop still walks the weighted training row set and
does membership checks against the dense column; only 2.7% of 200k nnz is dense
enough to trigger it, while the total-maintenance/update overhead and branch
costs affect the hot solve. This closes the complement-kernel item for the
current PLP workload.

After reverting the experiment and reinstalling, the clean 50k sanity check was:

```text
50k sparse-residual: 2.333s, variance 0.02600871, nonzero 131, KKT 1.542626
```

- Added experimental env-gated weighted update-row kernel:

```text
CYCLOPS_WEIGHTED_UPDATE_ROWS=1
```

  During CV, gradient/Hessian already scans fold-weighted training rows, but
  updateSparseLogisticState historically updated every row in the full column.
  The experimental path updates only nonzero-weight training rows. Validation
  scoring recomputes `X beta` from beta afterward, so the scored solution stays
  exact.

- Clean A/B in current worktree, 200k exact 3-fold CV, `threads=3`:

```text
toggle off: 22.390s, variance 0.01154783, nonzero 241,
            objective -10144.39, loglik -9833.856, KKT 100.17
toggle on:  21.531s, variance 0.01154783, nonzero 241,
            objective -10144.39, loglik -9833.856, KKT 100.17

update nnz:      6.288B -> 4.294B (68.3%)
update seconds:  5.886s -> 3.810s (64.7%)
wall speedup:    1.04x in this A/B run
```

Conclusion: weighted update rows are semantically promising and reduce update
work materially, but they are only an incremental wall-clock win because the
dominant gradient/Hessian scans are unchanged.

Follow-up, current build:

The weighted update-row path was promoted from opt-in to default `auto` mode.
The behavior is:

```text
CYCLOPS_WEIGHTED_UPDATE_ROWS unset:
    use filtered update rows only when nonzero-weight rows are a strict subset
    of the current data object

CYCLOPS_WEIGHTED_UPDATE_ROWS=0:
    disable filtered update rows

CYCLOPS_WEIGHTED_UPDATE_ROWS=1:
    force filtered update rows, for diagnostics
```

This keeps the exact improvement for global weighted CV while avoiding the
forced-mode slowdown on compact fold-local data, where the physical data object
already contains only training rows.

Exact auto-CV, `folds=3`, `threads=3`, same folds/tolerance/objective:

| sample | path | elapsed sec | variance | nonzero | KKT | CCD epochs | GH nnz | update nnz | GH sec | update sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | no fold-local, before auto update rows | 4.638 | 0.02747748 | 140 | 1.609 | 1176 | 8.741B | 1.365B | 7.130 | 1.301 |
| 50k | no fold-local, auto update rows | 4.319 | 0.02747748 | 140 | 1.609 | 1176 | 8.741B | 0.948B | 7.009 | 0.841 |
| 50k | fold-local default lazy, before auto update rows | 2.723 | 0.02600871 | 131 | 1.543 | 672 | 5.112B | 0.523B | 4.282 | 0.506 |
| 50k | fold-local default lazy + auto update rows | 2.585 | 0.02600871 | 131 | 1.543 | 672 | 5.112B | 0.523B | 4.157 | 0.480 |
| 200k | no fold-local, before auto update rows | 13.267 | 0.01154783 | 241 | 100.173 | 813 | 23.814B | 6.288B | 19.810 | 5.745 |
| 200k | no fold-local, auto update rows | 12.017-12.378 | 0.01154783 | 241 | 100.173 | 813 | 23.814B | 4.294B | 19.662-20.004 | 3.543-3.612 |
| 200k | fold-local default lazy, before auto update rows | 11.220 | 0.01228495 | 253 | 10.475 | 702 | 21.306B | 3.952B | 17.146 | 3.326 |
| 200k | fold-local default lazy + auto update rows | 11.058-11.197 | 0.01228495 | 253 | 10.475 | 702 | 21.306B | 3.952B | 16.683-17.070 | 3.242-3.276 |

Current retained speedups:

```text
50k no-fold-local update-row speedup:       4.638 / 4.319 = 1.07x
200k no-fold-local update-row speedup:     13.267 / 12.378 = 1.07x
50k fold-local combined vs no-fold-local:   4.638 / 2.585 = 1.79x
200k fold-local combined vs no-fold-local: 13.267 / 11.058 = 1.20x
```

Conclusion: default-auto update filtering is worth keeping because it is exact,
low risk, and helps the global weighted CV path. It does not materially change
the current goal status on 200k because the dominant cost remains
gradient/Hessian `fitted[row]` gathers.

Additional micro-kernel test, 2026-06-11:

- Added an env-gated software prefetch experiment for indicator-column
  gradient/Hessian and update loops:

```text
CYCLOPS_SPARSE_PREFETCH_DISTANCE=16
```

- Result on 200k exact 3-fold auto-CV, `threads=3`:

```text
same build, prefetch disabled: elapsed 22.181s, GH 23.851s, update 5.950s
prefetch distance 16:          elapsed 22.256s, GH 23.491s, update 5.956s
```

- The hook left an extra branch in the innermost loop when disabled, and
  prefetch distance 16 did not improve wall time. Removed the experiment from the
  code path rather than keeping a slower default. A future prefetch attempt should
  use separate specialized loop bodies so the no-prefetch path has no inner-loop
  branch, and should be tested with per-bucket microbenchmarks before any
  production default.

Additional micro-kernel test, 2026-06-11:

- Split the indicator-column gradient/Hessian loop into a direct unit-weight path
  and a non-unit weighted path. This targets CV training folds, where the
  fold-filtered rows usually have nonzero weights equal to 1.
- A first lambda-based helper version was slower and was replaced by straight-line
  branches.
- Result on 200k exact 3-fold auto-CV, `threads=3`, `sparse-residual`:

```text
lambda helper split: elapsed 21.722s, GH 21.650s, update 5.908s
direct branch split: elapsed 21.241s, GH 19.998s, update 5.790s
```

- The direct branch preserved the selected variance, nonzero count, objective
  difference, and final KKT check relative to the same-kernel reference:

```text
variance: 0.0115478318
nonzero:  241
KKT:      100.173
```

- This is at best a small/marginal wall-clock result. It may reduce measured
  gradient/Hessian time in this run, but it does not move end-to-end auto-CV
  enough to count toward the 50% goal.

Combined toggle check, 2026-06-11:

```text
CYCLOPS_FOLD_LOCAL_CV=1
CYCLOPS_WEIGHTED_UPDATE_ROWS=1
direct unit-weight GH branch
```

On 200k exact 3-fold auto-CV, `threads=3`:

```text
elapsed:              20.636s
variance:             0.0122849480
nonzero:              253
KKT:                  10.475
fold-local build sec: 5.628
GH sec:               17.174
update sec:            3.397
GH nnz:               21.306B
update nnz:            3.952B
```

This did not improve over the earlier fold-local-only result:

```text
fold-local-only elapsed: 19.414s
fold-local-only GH sec:  16.331s
fold-local-only update:   3.196s
```

Conclusion: do not treat the unit-weight branch or the combined toggle as a major
optimization. Keep only if later repeated medians show a stable GH reduction with
no wall-clock regression; otherwise drop it from the PR-ready branch.

## Step 7: Active-set IRLS/WLS warm accelerator

Do not retry all-column WLS/IRLS. It was slower because it scanned too much.

If implementing IRLS/WLS, only use it as an active-set warm accelerator:

1. Use strong rules to get candidate set S.
2. Run fitted-response CCD on S until reasonably stable.
3. Define A = nonzeros union recent KKT violators union top gradient candidates.
4. Build logistic quadratic approximation only on A.
5. Solve weighted L1 least-squares subproblem only on A.
6. Accept only with true logistic objective decrease.
7. Update `fitted[]` exactly.
8. Return to exact fitted-response CCD and full KKT repair.

The final accepted solution must still pass the exact Cyclops logistic KKT conditions.

This is high complexity and should come only after strong sets and KKT improvements.

### Active-set WLS/Gram cost diagnostic

Added a benchmark-only diagnostic:

```text
extras/benchmarks/diagnose_active_wls_gram.R
```

It fits one fixed training fold with the retained `sparse-residual` kernel, maps
the fitted coefficients back to the original matrix columns, then times the cost
of building an active-set weighted least-squares Gram matrix:

```text
eta = intercept + X beta
p   = sigmoid(eta)
w   = p * (1 - p)
z   = eta + (y - p) / max(w, 1e-8)

G_A = X_A' W X_A
b_A = X_A' W z
```

It also times 10 dense covariance-coordinate-descent sweeps on `G_A`, only as a
rough lower-level solve-cost proxy. This is diagnostic-only; it does not change
Cyclops solves.

#### Rounded active-size buckets

50k fold 1, variance `0.02600871`, fixed-fold CCD solve `0.167s`, support size
`102`:

| target A | active nnz share | Gram build sec | rhs sec | dense CD 10 epochs | Gram / CCD |
|---:|---:|---:|---:|---:|---:|
| 128 | 15.0% | 0.095 | 0.002 | 0.005 | 0.57x |
| 256 | 33.3% | 0.416 | 0.002 | 0.008 | 2.49x |
| 512 | 55.3% | 1.279 | 0.004 | 0.016 | 7.66x |
| 1024 | 75.5% | 2.451 | 0.005 | 0.035 | 14.68x |

200k fold 1, variance `0.0164943690178096`, fixed-fold CCD solve `1.566s`,
support size `224`:

| target A | active nnz share | Gram build sec | rhs sec | dense CD 10 epochs | Gram / CCD |
|---:|---:|---:|---:|---:|---:|
| 128 | 8.6% | 0.237 | 0.003 | 0.006 | 0.15x |
| 256 | 20.7% | 1.079 | 0.006 | 0.009 | 0.69x |
| 512 | 49.7% | 5.851 | 0.013 | 0.021 | 3.74x |
| 1024 | 74.3% | 12.905 | 0.019 | 0.048 | 8.24x |

#### Focused support-size checks

50k:

| target A | active nnz share | Gram+rhs+CD sec | total / CCD |
|---:|---:|---:|---:|
| 102 | 9.1% | 0.053 | 0.32x |
| 150 | 19.1% | 0.151 | 0.92x |
| 224 | 29.5% | 0.327 | 1.99x |

200k:

| target A | active nnz share | Gram+rhs+CD sec | total / CCD |
|---:|---:|---:|---:|
| 224 | 13.8% | 0.560 | 0.36x |
| 300 | 27.9% | 1.895 | 1.23x |
| 384 | 38.2% | 3.194 | 2.07x |

Interpretation:

- Dense Gram solve time is tiny; Gram construction dominates.
- Exact-support Gram can be cheaper than a fixed-fold CCD solve, especially on
  200k.
- The cost becomes unattractive as soon as the active set is expanded with hot
  columns. On 200k, going from 224 to 300 columns already costs more than the full
  fixed-fold fitted-response CCD solve.
- Therefore a Gram/WLS accelerator is only plausible if the active set is very
  close to the true support and repairs are rare. It is not a good broad
  replacement for fitted-response CCD, and it is risky for auto-CV strong sets
  where candidate sets often become large.

Recommendation: do not implement a general active-set Gram IRLS solver next. If
this path is revisited, make it a narrow late-path/final-fit accelerator with a
strict active-size/active-nnz budget and immediate fallback to fitted-response CCD
when `activeNnzShare` exceeds roughly 15-20%.

### Sparse WLS residual-CD IRLS diagnostic

Added a second benchmark-only IRLS/WLS diagnostic:

```text
extras/benchmarks/diagnose_sparse_wls_irls_cd.R
```

This is closer to glmnet's GLM shape than the diagonal row-pass shortcut. For one
training fold it:

1. builds compact CSC once;
2. forms an IRLS weighted least-squares problem;
3. solves the fixed-WLS L1 subproblem with sparse residual coordinate descent;
4. recomputes the exact logistic objective and KKT conditions after each outer
   step.

The inner WLS update is:

```text
r = z - intercept - X beta

intercept += sum_i w_i r_i / sum_i w_i
r_i       -= intercept_delta

for each column j:
    h_j   = sum_i w_i x_ij^2
    rho_j = sum_i w_i x_ij r_i + h_j beta_j
    beta_new = soft_threshold(rho_j, lambda) / h_j
    r_i -= x_ij * (beta_new - beta_old)
```

This remains diagnostic-only. It is not production Cyclops code and does not
change the retained fitted-response kernel.

50k fold 1, variance `0.02600871`, `threads=3`:

| method | setup/build sec | solve sec | final KKT | final nonzero |
|---|---:|---:|---:|---:|
| current fitted-response fixed fold | - | 0.151 | exact | 102 |
| sparse WLS residual-CD, 8 x 10 epochs | 0.089 | 0.723 | 8.345 | 102 |
| sparse WLS residual-CD, 26 x 20 epochs | 0.090 | 4.351 | 9.44e-7 | 104 |

200k fold 1, variance `0.01228495`, `threads=3`:

| method | setup/build sec | solve sec | final KKT | final nonzero |
|---|---:|---:|---:|---:|
| current fitted-response fixed fold | - | 1.249 | exact | 182 |
| sparse WLS residual-CD, 12 x 10 epochs | 0.374 | 5.449 | 9.897 | 184 |
| sparse WLS residual-CD, 30 x 20 epochs | 0.370 | 25.037 | 0.0130 | 180 |

Interpretation:

- This real WLS residual-CD variant is much better than the diagonal/Jacobi
  row-pass solver. It keeps support near the fitted-response support and converges
  on 50k.
- It is still far too slow as a full-feature replacement. On 50k it needs about
  `4.35 / 0.151 = 28.8x` the current fixed-fold fitted-response solve time to
  reach KKT. On 200k it is still not exact after `25.0 / 1.249 = 20.0x` the
  current fixed-fold solve time.
- The problem is not only implementation overhead: each WLS outer step needs many
  full sparse coordinate epochs, and repeated WLS subproblem solves cost more than
  direct fitted-response CCD on this PLP workload.
- Do not implement all-feature IRLS/WLS sparse residual CD in production.
- A glmnet-like route would need stronger path machinery than this diagnostic:
  small strong/ever-active sets, very few IRLS outer steps, and exact
  fitted-response KKT repair. Without a small active set, WLS/CD is not a speed
  path here.

## Benchmark protocol

For each major change, run at least:

- 50k exact 3-fold CV, threads = 3
- 200k exact 3-fold CV, threads = 3
- larger samples when extraction/runtime allow, threads = nFolds

If time permits, use 3 repetitions and report medians.

For each run, report:

- elapsed time
- speedup versus current fitted-response baseline
- selected variance/prior
- objective
- objective difference versus baseline
- CV score
- number of nonzeros
- prediction RMSE versus baseline
- max absolute prediction difference versus baseline
- max KKT violation
- RSS / peak memory if available

For performance counters, report normalized metrics:

- cycles / nnz scanned
- instructions / nnz scanned
- L1D misses / nnz scanned
- LLC misses / nnz scanned
- dTLB misses / nnz scanned

## Correctness rules

Do not merge an optimization unless:

- final solution passes exact KKT repair
- objective is equal to baseline within expected numerical tolerance
- predictions are equal or negligibly different
- selected prior/variance is unchanged, or any difference is explained by numerical tie behavior
- no safe-screened variable violates final KKT
- coefficient expansion after duplicate compression preserves predictions and objective

## Final report expected

Produce a concise report with:

1. Implemented changes.
2. 50k and 200k benchmark table.
3. Correctness validation table.
4. Perf-counter summary.
5. Explanation of accepted changes.
6. Rejected experiments and why.
7. Remaining risks.
8. Recommended PR structure.

Preferred PR sequence:

1. instrumentation and counters
2. strong-rule active sets with exact KKT repair
3. row-wise KKT scanner
4. duplicate compression if useful
5. Gap Safe screening if useful
6. fold-local CSC / micro-kernel experiments if useful

Keep changes small enough that each PR can be reviewed and benchmarked independently.

## GAP-safe critical-radius diagnostic

After the basic GAP-safe diagnostic screened many columns but almost no nnz, add a
diagnostic-only critical-radius trace behind:

```text
CYCLOPS_GAP_SAFE_RADIUS_TRACE=/path/to/trace.csv
```

This trace does not change the solve. For each GAP-safe check it reports
nnz-weighted screening curves at the current safe radius and at smaller hypothetical
radii:

```text
R, R/2, R/4, R/8, R/16, R/32, R/64, and R = 0
```

It reports both raw feature norms and intercept-centered norms. The centered norm
is only a diagnostic approximation for the intercept equality; it is not used for
production screening.

### 200k exact CV, stdstep 2, folds 3, threads 3

Run sequentially, one kernel at a time, on:

```text
ducky-dementia-plp-sample-200000.rds
rows=199873 features=16676 nnz=42952980 outcomes=2267
```

With diagnostics enabled:

| kernel | elapsed sec | selected variance | nonzero |
|---|---:|---:|---:|
| sparse-residual | 28.291 | 0.01649437 | 298 |
| sparse-residual-strong | 25.268 | 0.01590367 | 317 |

The elapsed times include the extra GAP-safe diagnostic passes and should not be
treated as solver timings.

#### Sparse-residual radius curve

| radius | implied gap factor | screened nnz share | screened hot-nnz share | centered screened nnz share | centered hot-nnz share |
|---|---:|---:|---:|---:|---:|
| R | 1 | 0.0256% | 0.0000% | 0.0256% | 0.0000% |
| R/2 | 1/4 | 0.1102% | 0.0000% | 0.1102% | 0.0000% |
| R/4 | 1/16 | 0.4299% | 0.0000% | 0.4311% | 0.0000% |
| R/8 | 1/64 | 1.4307% | 0.0000% | 1.4367% | 0.0000% |
| R/16 | 1/256 | 4.2231% | 0.1538% | 4.2567% | 0.1714% |
| R/32 | 1/1024 | 10.1587% | 2.6016% | 10.4996% | 2.9864% |
| R/64 | 1/4096 | 21.4587% | 11.3056% | 22.8501% | 12.9825% |
| 0 | 0 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |

#### Sparse-residual-strong radius curve

| radius | implied gap factor | screened nnz share | screened hot-nnz share | centered screened nnz share | centered hot-nnz share |
|---|---:|---:|---:|---:|---:|
| R | 1 | 0.0138% | 0.0000% | 0.0138% | 0.0000% |
| R/2 | 1/4 | 0.0650% | 0.0000% | 0.0651% | 0.0000% |
| R/4 | 1/16 | 0.2723% | 0.0000% | 0.2725% | 0.0000% |
| R/8 | 1/64 | 1.0087% | 0.0000% | 1.0106% | 0.0000% |
| R/16 | 1/256 | 3.1851% | 0.0000% | 3.2056% | 0.0000% |
| R/32 | 1/1024 | 8.6214% | 0.6614% | 8.7616% | 0.7841% |
| R/64 | 1/4096 | 19.0960% | 7.6601% | 20.3129% | 9.1120% |
| 0 | 0 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |

### Interpretation

The current radius screens the wrong columns for this PLP workload: many rare
columns, almost no nnz, and no hot nnz. Intercept-centering barely changes the
current result.

Hot-nnz screening does not appear until approximately `R/16` to `R/32`, which
corresponds to roughly 256x to 1024x smaller dual gaps. Meaningful total nnz
screening appears around `R/32` to `R/64`, or roughly 1024x to 4096x smaller gaps.

This argues against wiring the current basic sphere GAP-safe rule into the
production solver. The only safe-screening route worth revisiting later would be a
substantially tighter celer-style dual-extrapolation/working-set method. The next
more direct optimization should be fold-local compact CSC/data-layout work.

## GAP-safe over-solve diagnostic

A staged warm-start over-solve diagnostic was added:

```text
extras/benchmarks/diagnose_gap_safe_oversolve.R
```

It runs one representative 200k training fold/lambda at normal tolerance, then
continues from the fitted coefficients with tighter tolerances. This is not full CV
and is intended only to answer whether a smaller dual gap appears cheaply enough to
screen hot nnz.

Run:

```text
kernel=sparse-residual
variance=0.0164943690178096
folds=3
fold-index=1
threads=3
tolerances=1e-6,1e-7,1e-8
```

Result:

| tolerance | elapsed sec | extra sec after normal | gap | radius | screened nnz share | screened hot nnz |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-6 | 6.539 | 0.000 | 533.879 | 16.338 | 0.000% | 0 |
| 1e-7 | 4.183 | 4.183 | 75.507 | 6.144 | 0.019% | 0 |
| 1e-8 | 7.011 | 11.194 | 0.986 | 0.702 | 1.481% | 0 |

This closes the basic/dynamic GAP-safe path for now. Even after spending more time
than the normal fold solve again, the current sphere rule still screens no hot nnz
and only 1.48% total eligible nnz.

## Fold-local compact CSC A/B prototype

A benchmark-only fold-local CSC prototype was added:

```text
extras/benchmarks/benchmark_fold_local_csc.R
```

It compares:

```text
weighted-view:
    full global matrix
    0/1 training weights
    global fitted[] length
    fold-weighted column views

compact-csc:
    physical training-row subset
    local row ids 0..n_train-1
    compact fitted[] length
    no validation rows in hot loops
```

This is not yet production CV integration. It is an A/B layout experiment.

### 200k fixed-lambda fold experiment

Run:

```text
matrix=ducky-dementia-plp-sample-200000.rds
rows=199873
features=16676
nnz=42952980
variance=0.0164943690178096
folds=3
threads=3
tolerance=1e-6
kernel=sparse-residual
```

| fold | layout | build sec | solve sec | total sec | training nnz | nonzero |
|---:|---|---:|---:|---:|---:|---:|
| 1 | weighted-view | 4.687 | 4.886 | 4.886 | 28,569,430 | 224 |
| 1 | compact-csc | 2.810 | 3.221 | 6.031 | 28,569,430 | 224 |
| 2 | weighted-view | 0.000 | 5.060 | 5.060 | 28,689,951 | 211 |
| 2 | compact-csc | 2.679 | 3.369 | 6.048 | 28,689,951 | 211 |
| 3 | weighted-view | 0.000 | 3.545 | 3.545 | 28,646,579 | 230 |
| 3 | compact-csc | 2.789 | 2.270 | 5.059 | 28,646,579 | 230 |

Solve totals:

| layout | total solve sec |
|---|---:|
| weighted-view | 13.491 |
| compact-csc | 8.860 |

Compact CSC solves were 1.50x to 1.56x faster. The mean gradient/Hessian time per
billion scanned nnz improved from 1.116 seconds to 0.759 seconds. Update time per
billion update nnz improved from 0.928 seconds to 0.854 seconds.

However, the initial R-level materialization cost was about 2.7-2.8 seconds per
fold. If paid naively per fold, build cost erases the solve-time saving on this
200k fixed-lambda prototype. The result is still promising for production work
because the hot-loop solve speed improves clearly; the production question is
whether fold-local CSC can be built cheaply in C++ and reused across lambda path
solves.

### C++ compact-row materialization test

A benchmark-only Rcpp helper was added:

```text
.cyclopsCompactRowsDgCMatrix(matrix, keepRows)
```

It compacts a `dgCMatrix` by rows in C++:

```cpp
rowMap[oldRow] = newRow or -1

for each column j:
    count kept nonzeros
    build new column pointer p[j + 1]

for each column j:
    for each old nonzero k:
        newRow = rowMap[i[k]]
        if newRow >= 0:
            newI[out] = newRow
            newX[out] = x[k]
            ++out
```

The solver is unchanged. The compact matrix is still converted back through the
normal Cyclops data-construction path, so this tests only whether row compaction
itself is cheap enough.

Run:

```text
extras/benchmarks/benchmark_fold_local_csc.R
  --matrix-rds ducky-dementia-plp-sample-200000.rds
  --kernel sparse-residual
  --variance 0.0164943690178096
  --folds 3
  --threads 3
  --compact-build-method cpp
```

Result:

| fold | layout | build sec | compact matrix sec | Cyclops data build sec | solve sec | total sec | training nnz | nonzero |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | weighted-view | 6.280 | 0.000 | 6.280 | 6.389 | 6.389 | 28,569,430 | 224 |
| 1 | compact-csc | 4.305 | 0.535 | 3.770 | 4.654 | 8.959 | 28,569,430 | 224 |
| 2 | weighted-view | 0.000 | 0.000 | 0.000 | 5.473 | 5.473 | 28,689,951 | 211 |
| 2 | compact-csc | 2.872 | 0.352 | 2.520 | 3.274 | 6.146 | 28,689,951 | 211 |
| 3 | weighted-view | 0.000 | 0.000 | 0.000 | 3.426 | 3.426 | 28,646,579 | 230 |
| 3 | compact-csc | 2.680 | 0.322 | 2.358 | 2.496 | 5.176 | 28,646,579 | 230 |

Totals:

| layout | build sec | compact matrix sec | Cyclops data build sec | solve sec | total sec |
|---|---:|---:|---:|---:|---:|
| weighted-view | 6.280 | 0.000 | 6.280 | 15.288 | 15.288 |
| compact-csc | 9.857 | 1.209 | 8.648 | 10.424 | 20.281 |

This changes the diagnosis. C++ row compaction itself is cheap:

```text
1.209 seconds total across all 3 compact folds
about 0.014 seconds per million training nnz copied/remapped
```

The remaining build cost is the generic compact-matrix-to-Cyclops-data
construction:

```text
8.648 seconds total across all 3 compact folds
```

The compact solve remains materially faster:

```text
weighted-view solve total: 15.288s
compact-csc solve total:  10.424s
solve speedup:             1.47x
```

but the benchmark-only prototype is still slower end-to-end for one fixed lambda
because it rebuilds full Cyclops data objects for each fold.

### Native fold-local Cyclops data test

A second benchmark-only path was added:

```text
.cyclopsMakeFoldLocalData(cyclopsData, keepRows)
```

This bypasses the R `dgCMatrix` round-trip and generic Cyclops conversion. It
builds a new native Cyclops `ModelData` object directly from the existing full
Cyclops data object:

```cpp
rowMap[oldRow] = newRow or -1

compact y / type / time / pid by kept rows

for each Cyclops column:
    if dense:
        copy values[keptRows]
    if sparse:
        copy/remap kept row ids and values
    if indicator:
        copy/remap kept row ids only
    if intercept:
        copy intercept marker

preserve column numeric labels
preserve intercept/offset/finalized flags
```

The benchmark script now has:

```text
--layouts native-fold-csc
--layouts all
```

This is still not production CV integration. It proves the native fold-local data
path and measures the build/solve tradeoff.

#### 200k all-layout run

Run:

```text
extras/benchmarks/benchmark_fold_local_csc.R
  --matrix-rds ducky-dementia-plp-sample-200000.rds
  --kernel sparse-residual
  --variance 0.0164943690178096
  --folds 3
  --threads 3
  --layouts all
  --compact-build-method cpp
```

Totals:

| layout | build sec | solve sec | total sec |
|---|---:|---:|---:|
| weighted-view | 13.339 | 39.460 | 39.460 |
| compact-csc | 30.700 | 25.500 | 56.200 |
| native-fold-csc | 1.645 | 22.538 | 24.183 |

The old compact `dgCMatrix` route still pays most of its time in generic Cyclops
data construction. The native route removes that cost:

```text
compact-csc build total:      30.700s
native-fold-csc build total:   1.645s
```

Native fold-local also preserves the compact-layout solve benefit:

```text
weighted-view solve total:    39.460s
native-fold-csc solve total:  22.538s
solve speedup:                 1.75x
```

Fold-level nonzero counts matched exactly across layouts:

```text
fold 1: 224
fold 2: 211
fold 3: 230
```

The native and compact `dgCMatrix` paths had identical log-likelihood differences
relative to weighted-view, confirming that the native clone is reproducing the
same compact training problem as the R materialization route.

#### 200k focused A/B run

Run:

```text
--layouts weighted-view,native-fold-csc
```

Totals:

| layout | build sec | solve sec | total sec |
|---|---:|---:|---:|
| weighted-view | 13.084 | 34.265 | 34.265 |
| native-fold-csc | 1.749 | 23.543 | 25.292 |

The focused run gives the cleaner fixed-lambda fold-replay estimate:

```text
solve speedup:      34.265 / 23.543 = 1.46x
total fold speedup: 34.265 / 25.292 = 1.35x
```

Native build cost is now small enough to amortize across a CV path:

```text
native build total across 3 folds: 1.749s
native build mean per fold:        0.583s
```

For production auto-CV, each fold-local object should be built once and reused
across all variance/lambda candidates. That should make the fixed build cost much
less important than in this one-fixed-lambda benchmark.

### Layout metrics

| layout | fitted bytes | max row id touched | mean row id touched |
|---|---:|---:|---:|
| weighted-view | 1,598,984 | ~199,872 | ~106,500 |
| compact-csc | 1,065,984-1,065,992 | ~133,248 | ~71,000 |

The compact fold reduces fitted-state footprint by about one third, matching the
2/3 training fraction of 3-fold CV.

### Perf sample

One process-level `perf stat` sample was collected on fold 1 for each layout. This
includes R/Cyclops materialization around the fit, so it is less clean than the
internal Cyclops hot-loop counters above.

| layout | solve sec | build sec | cycles / GH nnz | instructions / GH nnz | L1D misses / GH nnz | LLC misses / GH nnz | dTLB misses / GH nnz |
|---|---:|---:|---:|---:|---:|---:|---:|
| weighted-view | 4.857 | 4.766 | 32.58 | 68.83 | 0.983 | 0.0199 | 0.00655 |
| compact-csc | 3.312 | 3.095 | 24.16 | 53.11 | 0.785 | 0.0233 | 0.00500 |

The process-level sample supports fewer cycles/instructions per scanned nnz for the
compact layout, but LLC rates are noisy because the sample includes build and R
work. Treat the internal gradient/Hessian seconds per nnz as the cleaner signal.

### Fold-local CSC decision

Keep pursuing fold-local compact CSC, but do not merge a naive R-style
materialization strategy. The C++ row-copy test shows that direct row compaction is
not the expensive part; rebuilding generic Cyclops data objects is. The native
fold-local test shows that direct Cyclops `ModelData` cloning fixes that setup
problem in a benchmark path. The next production prototype should:

1. Move the native fold-local builder from benchmark helper into the CV setup path.
2. Build each fold once and reuse it across all lambdas in the CV path.
3. Avoid duplicating unnecessary value arrays for indicator columns where ownership
   and row-remapping semantics allow it.
4. Report row-compaction, model-data build, and solve time separately.
5. Keep the current weighted-view path as fallback when memory or build cost is too
   high.

### Production CV fold-local prototype status

The native fold-local builder was wired into the CV path behind:

```text
CYCLOPS_FOLD_LOCAL_CV=1
```

It builds one compact training CCD per fold and reuses it across auto-CV
candidates. Scoring is done on the original full-data CCD after copying the compact
training beta back, so validation scoring uses the same full-data row space as the
weighted-view baseline.

Implementation/correctness fixes made during the production prototype:

1. Added fold-local build counters to verify whether compact clones are actually
   used.
2. Fixed validation-weight construction bug in an earlier validation-clone attempt:
   `CrossValidationSelector::getComplement()` mutates an existing weights vector and
   does not initialize/resize an empty vector.
3. Fixed sparse logistic predictive log-likelihood when `sparseLogisticXBetaDirty`
   is true: the old sparse branch added case log-odds and subtracted denominators
   only for case rows; it now subtracts the denominator term for all weighted rows.
4. Changed compact scoring to copy compact beta back to the full original CCD and
   score there, avoiding compact validation clones.
5. Renumbered fold-local compact pids contiguously in the row-subset builder. This
   removes old global pid ids from compact folds and avoids oversized patient-level
   arrays.
6. Added `updateWarmStartStateFrom()` and copied coordinate step bounds (`hDelta`)
   along with beta before compact fold solves. This tested whether baseline/compact
   divergence came from warm-start bound state.

Current 200k exact 3-fold auto-CV, `threads=3`, same installed build:

| path | elapsed sec | selected variance | nonzero | final KKT check | build attempts | build successes | build sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| weighted-view baseline | 21.357 | 0.0115478318 | 241 | 100.173 | 0 | 0 | 0.000 |
| fold-local compact | 19.414 | 0.0122849480 | 253 | 10.475 | 3 | 3 | 5.418 |

Hot-loop counters from the same run:

| path | GH nnz | GH sec | update nnz | update sec | CCD epochs |
|---|---:|---:|---:|---:|---:|
| weighted-view baseline | 23.814B | 22.699 | 6.288B | 5.641 | 813 |
| fold-local compact | 21.306B | 16.331 | 3.952B | 3.196 | 702 |

This means the compact layout is doing useful work inside the kernel:

```text
wall-clock speedup:       21.357 / 19.414 = 1.10x
gradient/Hessian speedup: 22.699 / 16.331 = 1.39x
update speedup:            5.641 /  3.196 = 1.76x
```

However, live auto-CV still selects a different variance than the weighted-view
baseline. The pid renumbering and `hDelta` warm-start sync did not change that
result, so those were not the root cause.

Replay using the same first three candidate variances from the baseline trace gives
matching final refits:

| path | elapsed sec | final replay variance | nonzero | final KKT check |
|---|---:|---:|---:|---:|
| weighted-view replay | 25.036 | 0.0134247999 | 266 | 2.819 |
| fold-local replay | 21.368 | 0.0134247999 | 266 | 2.819 |

The replay wall-clock speedup is:

```text
25.036 / 21.368 = 1.17x
```

Fold-level replay traces show the compact folds often converge to smaller full-KKT
violations than the weighted-view folds, so the predictive fold scores differ even
for identical candidate variances. Example at the first candidate:

| fold | baseline fold KKT | compact fold KKT | pred log-likelihood diff |
|---|---:|---:|---:|
| 1 | 48.98 | 3.58 | -12.69 |
| 2 | 75.81 | 6.47 | -10.75 |
| 3 | 59.03 | 1.74 | -10.91 |

Interpretation: the compact layout is not merely a row-id locality change in live
auto-CV; because it changes the amount of work and convergence trajectory per fold,
the default gradient-convergence auto-CV scores can shift. For fixed replayed
candidates, the final full-data refit agrees, but production auto-CV equivalence is
not yet clean enough to treat fold-local as a drop-in optimization.

Additional fixed-fold tolerance diagnostic:

```text
200k fold 1, variance 0.0164943690178096, tolerance 1e-8, threads=3

weighted-view:    solve 6.167s, log likelihood -6474.003, nonzero 217
native-fold-csc:  solve 7.091s, log likelihood -6473.951, nonzero 214
```

Tightening tolerance makes the fold scores closer, but the compact fold is slower
and still does not match support exactly. This supports the diagnosis that
fold-local compact CSC is not suffering from an obvious row-subset data bug; the
remaining issue is solver path/convergence interaction. Tightening fold solves is
not an acceptable performance strategy for the current goal.

Decision: keep fold-local compact CSC as a promising but not production-ready
layout experiment. It gives clear kernel-level speedups, but only ~1.10-1.17x
end-to-end so far and does not meet the 50% stable speedup goal. The next
productive work should either make fold-level convergence/search-path equivalence
strict enough for production, or move to another kernel/layout idea with larger
expected wall-clock impact.

#### Fold-local follow-up: row labels and process perf

Tried skipping row-label string copies in `createRowSubsetFromBinaryWeights()` for
internal fold-local CV clones. This did not improve the measured build sum or wall
time on 200k:

```text
before label skip: elapsed 19.414s, foldLocalBuildSeconds 5.418
after label skip:  elapsed 19.953s, foldLocalBuildSeconds 5.487
```

Conclusion: row labels are not the important fold-local build cost. The remaining
build cost is likely dominated by column remapping/allocation and model-specific
object construction.

Process-level `perf stat` on 200k exact 3-fold auto-CV, `threads=3`, same build:

| path | elapsed sec | core cycles | core instructions | core L1D misses | core LLC misses | core dTLB misses |
|---|---:|---:|---:|---:|---:|---:|
| weighted-view | 20.718 | 224.3B | 575.5B | 18.35B | 96.99M | 17.20M |
| fold-local | 20.376 | 206.2B | 513.0B | 13.79B | 138.93M | 16.92M |

Interpretation:

- fold-local reduced core cycles by ~8%, instructions by ~11%, and L1D misses by
  ~25%;
- LLC-load misses increased in this process-level sample, likely reflecting clone
  build/allocation and different search/convergence work mixed into the same
  process counters;
- internal Cyclops counters remain the cleaner signal for the hot loops, where
  fold-local reduced gradient/Hessian time ~1.3-1.4x and update time ~1.7x;
- the process-level wall gain remains too small for the current goal.

#### Fold-local build split and constructor check

Added split counters for production fold-local clone construction:

```text
sparseLogistic.foldLocalDataBuildSeconds
sparseLogistic.foldLocalSpecificsBuildSeconds
sparseLogistic.foldLocalCcdSetupSeconds
```

200k exact 3-fold auto-CV, `threads=3`, `CYCLOPS_FOLD_LOCAL_CV=1`:

```text
elapsed:                        20.259s
foldLocalBuildSeconds:           5.622s
foldLocalDataBuildSeconds:       0.760s
foldLocalSpecificsBuildSeconds:  0.000005s
foldLocalCcdSetupSeconds:        4.862s
GH seconds:                     17.060s
update seconds:                  3.380s
variance:                        0.0122849480
nonzero:                         253
KKT:                            10.475
```

This explains the mismatch between the benchmark-only native clone and production
CV: native row-subset data creation is already cheap in production. The expensive
part is CCD setup, mainly initializing fold-local state for the copied warm-start
beta.

Tried skipping constructor `computeXBeta()` when the copied beta vector is all
zero. On this 200k auto-CV run it did not fire materially:

```text
elapsed:                        20.309s
foldLocalBuildSeconds:           5.679s
foldLocalDataBuildSeconds:       0.771s
foldLocalSpecificsBuildSeconds:  0.000005s
foldLocalCcdSetupSeconds:        4.908s
GH seconds:                     17.459s
update seconds:                  3.366s
```

Inspection showed the simple zero-column concern is already handled:
`ModelSpecifics::axpyXBeta()` checks `beta != 0.0` before scanning a column.
Therefore the setup cost is not caused by scanning every zero column; it is real
work over nonzero warm-start columns and sparse logistic state synchronization.

Potential next fold-local improvement: avoid recomputing fold-local `X beta` from
columns during clone setup by copying/subsetting the source row state (`xBeta` or
fitted probability) into the fold-local model when the source state is current.
That is more invasive because the sparse fitted-response state can be dirty with
respect to materialized `xBeta`, so it needs a careful model-specific API rather
than another small constructor guard.

Tried a narrow `X beta` row-state copy experiment:

```text
sourceXBeta = source.getXBeta()
foldLocal.hXBeta[i] = sourceXBeta[keptRows[i]]
foldLocal xBetaKnown = true
foldLocal sufficientStatisticsKnown = false
```

This was wired behind:

```text
CYCLOPS_FOLD_LOCAL_COPY_XBETA=1
```

and tested on 200k exact 3-fold auto-CV, `threads=3`,
`CYCLOPS_FOLD_LOCAL_CV=1`:

```text
elapsed:                        22.280s
variance:                        0.01433457
nonzero:                         273
KKT:                              2.860
foldLocalBuildSeconds:           5.513s
foldLocalDataBuildSeconds:       0.729s
foldLocalSpecificsBuildSeconds:  0.000004s
foldLocalCcdSetupSeconds:        4.784s
GH seconds:                     20.901s
update seconds:                  3.982s
CCD epochs:                      875
```

Conclusion: do not enable this by default. It did not reduce setup time, because
materializing `source.getXBeta()` moved comparable work into the clone setup
window, and it changed the auto-CV search path. A useful state-copy optimization
would need to copy the already-current fitted-response state directly without
forcing source `X beta` materialization, and would need stricter fold-score
equivalence diagnostics.

After gating the experiment off by default, the normal fold-local path returned to
the earlier behavior:

```text
elapsed:                        19.949s
variance:                        0.0122849480
nonzero:                         253
KKT:                             10.475
foldLocalBuildSeconds:           5.546s
foldLocalDataBuildSeconds:       0.744s
foldLocalSpecificsBuildSeconds:  0.000006s
foldLocalCcdSetupSeconds:        4.802s
GH seconds:                     16.835s
update seconds:                  3.274s
CCD epochs:                      702
```

Same installed build, sequential weighted-view A/B:

| path | elapsed sec | variance | nonzero | KKT | CCD epochs | GH sec | update sec | GH nnz | update nnz | build sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weighted-view | 20.682 | 0.0115478318 | 241 | 100.173 | 813 | 20.238 | 5.682 | 23.814B | 6.288B | 0.000 |
| fold-local | 19.949 | 0.0122849480 | 253 | 10.475 | 702 | 16.835 | 3.274 | 21.306B | 3.952B | 5.546 |

Current-build wall speedup is only:

```text
20.682 / 19.949 = 1.04x
```

The hot-loop savings are real, but setup cost and auto-CV path divergence still
dominate the end-to-end result.

Tried a cheaper fitted-response state copy experiment behind:

```text
CYCLOPS_FOLD_LOCAL_COPY_FITTED=1
```

Instead of forcing `source.getXBeta()`, this copies the source sparse logistic
fitted probabilities when the source sparse state is already ready:

```text
foldLocal.fitted[i] = source.fitted[keptRows[i]]
foldLocal sparseLogisticStateReady = true
foldLocal sparseLogisticXBetaDirty = true
```

The normal lazy weight/statistics path can then materialize compact row state from
`fitted[]` in O(K) when needed, avoiding a nonzero-column `X beta` recompute when
the copied state is usable.

Current exact auto-CV results, same installed build:

| sample | path | elapsed sec | variance | nonzero | KKT | fold build sec | setup sec | GH sec | update sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | weighted-view | 6.108 | 0.02747748 | 140 | 1.609 | 0.000 | 0.000 | 6.997 | 1.281 |
| 50k | fold-local | 4.744 | 0.02600871 | 131 | 1.543 | 1.335 | 1.142 | 4.177 | 0.482 |
| 50k | fold-local + copy-fitted | 4.675 | 0.02600871 | 131 | 1.543 | 1.261 | 1.073 | 4.180 | 0.474 |
| 200k | weighted-view | 20.682 | 0.0115478318 | 241 | 100.173 | 0.000 | 0.000 | 20.238 | 5.682 |
| 200k | fold-local | 19.949 | 0.0122849480 | 253 | 10.475 | 5.546 | 4.802 | 16.835 | 3.274 |
| 200k | fold-local + copy-fitted | 19.316 | 0.0122849480 | 253 | 10.475 | 5.471 | 4.745 | 16.256 | 3.227 |

Speedups:

```text
50k copy-fitted vs weighted-view: 1.31x
50k copy-fitted vs fold-local:    1.01x
200k copy-fitted vs weighted-view: 1.07x
200k copy-fitted vs fold-local:    1.03x
```

Later diagnostic counters found that this interpretation was wrong. The first
copy-fitted experiment had not actually copied fitted state: the fold-local row
mapping was only passed to the owned CCD constructor when
`CYCLOPS_FOLD_LOCAL_COPY_XBETA=1`, so `CYCLOPS_FOLD_LOCAL_COPY_FITTED=1` alone
could not initialize compact fitted probabilities.

After fixing the constructor call to always pass `keptRows`, the rerun showed
that the source sparse logistic state was not ready at clone time:

| sample | path | elapsed sec | variance | nonzero | KKT | copy attempts | source ready | copy successes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 50k | fold-local + copy-fitted fixed | 4.539 | 0.02600871 | 131 | 1.543 | - | - | - |
| 200k | fold-local + copy-fitted fixed | 19.881 | 0.0122849480 | 253 | 10.475 | - | - | - |
| 200k | fold-local + copy-fitted counted | 19.841 | 0.0122849480 | 253 | 10.475 | 3 | 0 | 0 |

Tried refreshing the source fitted-response state before clone when possible:

```text
if copy-fitted && source.xBetaKnown && !source.hasSparseLogisticStateReady():
    source.refreshSparseLogisticState(false)
```

This did not make the copy fire:

| sample | path | elapsed sec | variance | nonzero | KKT | copy attempts | source ready | copy successes | fold build sec | GH sec | update sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200k | fold-local + copy-fitted refresh | 19.201 | 0.0122849480 | 253 | 10.475 | 3 | 0 | 0 | 5.434 | 16.314 | 3.246 |

Because `sourceReady` and `copySuccesses` were both zero, the apparent 19.201s
result should be treated as run-to-run noise, not a real copy-fitted speedup.

#### Sparse row-locality diagnostic

Added a benchmark-only layout diagnostic:

```text
extras/benchmarks/diagnose_sparse_layout_locality.R
```

It does not call the solver. It scans the PLP `dgCMatrix` and compares three row
id layouts:

```text
full-global:   original full matrix row ids
weighted-view: training rows selected by fold weights, but with global row ids
compact-fold:  same training rows remapped to 0..n_train-1
```

For each layout it reports column nnz buckets, row-id delta buckets, fitted-vector
cache-line reuse (`row / 8` for doubles), row-block touches/occupancy, and row
nnz distributions.

50k result, hot columns:

```text
weighted-view hot nnz share: ~57.5-57.8%
compact-fold hot nnz share:  ~57.5-57.8%

hot row-id transitions:
  weighted-view: delta<=7 ~72.8%, delta 8..63 ~27.0%, delta 64..511 ~0.14-0.15%
  compact-fold:  delta<=7 ~84.8%, delta 8..63 ~15.1%, delta 64..511 ~0.035-0.038%

hot cache-line reuse:
  weighted-view: ~1.94 fitted loads per touched cache line
  compact-fold:  ~2.54-2.55 fitted loads per touched cache line

hot 512-row block occupancy:
  weighted-view: ~18.7%
  compact-fold:  ~27.7-27.8%
```

200k result, hot columns:

```text
weighted-view hot nnz share: ~81.4-81.5%
compact-fold hot nnz share:  ~81.4-81.5%

hot row-id transitions:
  weighted-view: delta<=7 ~61.2%, delta 8..63 ~36.3%, delta 64..511 ~2.38%
  compact-fold:  delta<=7 ~73.3%, delta 8..63 ~25.6%, delta 64..511 ~1.07%

hot cache-line reuse:
  weighted-view: ~1.72 fitted loads per touched cache line
  compact-fold:  ~2.18 fitted loads per touched cache line

hot 512-row block occupancy:
  weighted-view: ~14.3%
  compact-fold:  ~21.5%

hot 4096-row block touches:
  weighted-view: ~48.9 blocks per hot column
  compact-fold:  ~32.9 blocks per hot column
```

The row distribution itself is stable across layouts:

```text
200k training folds:
  rows per compact fold: 133,248-133,249
  training nnz per fold: 28.57M-28.69M
  mean row nnz:          ~214-215
  median row nnz:        186
  p99 row nnz:           684-689
```

Interpretation:

- The measured compact-fold hot-loop speedup is explained by row-id locality:
  removing validation-row holes makes hot-column row ids denser, increases
  fitted-cache-line reuse, and reduces the number of row blocks touched.
- A simple cache-blocked CSC pass is unlikely to be a large additional win because
  row ids are already sorted within each column, and hot columns already touch
  almost every row block in the current row order.
- The remaining layout gains would need either a more compact row-index encoding
  for hot columns, row reordering that groups similar patients across columns, or a
  solver/kernel that loads `fitted[i]` once per row instead of once per nonzero.
- This supports keeping fold-local compact CSC as the best current layout idea,
  but it also explains why it has not delivered the required extra 50% end-to-end
  speedup by itself.

#### Rejected simple row-reordering experiment

Extended the benchmark-only fold-local CSC script with:

```text
--row-order none
--row-order nnz-desc
--row-order nnz-asc
--row-order outcome-nnz-desc
```

The row-order option applies only to the R `compact-csc` materialization path. It
does not change production Cyclops code. The goal was to test whether grouping
compact fold rows by row density or outcome could improve the remaining
`fitted[row]` locality enough to justify a native ordered-row fold-local builder.

Fixed-lambda fold checks, exact same training fold/objective/tolerance:

```text
50k fold 1, variance 0.02600871:
  compact-csc none:      solve 0.169s, nonzero 102
  compact-csc nnz-desc:  solve 0.171s, nonzero 102

200k fold 1, variance 0.0164943690178096:
  compact-csc none:              solve 1.536s, nonzero 224
  compact-csc nnz-desc:          solve 1.527s, nonzero 224
  compact-csc outcome-nnz-desc:  solve 1.558s, nonzero 224
```

Conclusion: simple row-density or outcome/density ordering is neutral to slightly
negative. Do not add native ordered-row support for these orderings. Any useful
row reordering would need a more structured similarity/blocking objective than
row nnz alone, and it is not the next highest-probability path.

#### Row-pass/CSR full-gradient diagnostic

Added a benchmark-only diagnostic:

```text
extras/benchmarks/diagnose_row_pass_gradient.R
```

It compiles a local Rcpp helper and compares two exact ways to compute one full
training-fold logistic gradient/Hessian pass for the sparse matrix:

```text
CSC column pass:
  for each feature j:
    for each row in column j:
      p = fitted[row]
      g[j] += x_ij * (p - y[row])
      h[j] += x_ij^2 * p * (1 - p)

CSR row pass:
  for each row i:
    p = fitted[i]
    q = p * (1 - p)
    for each feature j in row i:
      g_thread[j] += x_ij * (p - y[i])
      h_thread[j] += x_ij^2 * q
  reduce per-thread g/h arrays
```

This is diagnostic-only. It does not change the solver, and it does not imply CCD
can directly replace coordinate-wise scans with one row pass without changing the
optimization algorithm.

All runs used `threads=3`, fold 1 training rows, and three sequential repetitions.
Checksums matched exactly between methods.

| sample | method | median sec | training nnz | speedup vs CSC |
|---|---|---:|---:|---:|
| 50k | CSC column pass | 0.003106 | 7.159M | 1.00x |
| 50k | CSR row pass | 0.002832 | 7.159M | 1.10x |
| 200k | CSC column pass | 0.012459 | 28.569M | 1.00x |
| 200k | CSR row pass | 0.009875 | 28.569M | 1.26x |

One-time filtered layout build costs in the diagnostic:

| sample | CSC filtered build sec | CSR row build sec |
|---|---:|---:|
| 50k | 0.098 | 0.135 |
| 200k | 0.402 | 0.595 |

Interpretation:

- Loading `fitted[i]` once per row and scattering to per-thread gradient arrays is
  faster than a CSC full-gradient pass, especially on 200k.
- The raw advantage is modest, about 1.26x on the larger fold, before accounting
  for the CSR build/storage cost.
- This is not enough to justify a production solver rewrite by itself, because the
  current CCD algorithm needs updated `fitted[]` after each coordinate update.
- CSR row pass remains useful as a building block for a different algorithmic
  family: row-wise KKT/violation scoring, proximal/diagonal Newton steps, or
  IRLS/WLS setup. It is not a drop-in replacement for the current coordinate loop.

#### Rejected row-pass diagonal IRLS / proximal-Newton diagnostic

Added a benchmark-only solver diagnostic:

```text
extras/benchmarks/diagnose_diagonal_irls_rowpass.R
```

This tests the most direct row-pass alternative to coordinate descent. For one
training fold it builds compact CSR once, then iterates:

```text
for each row i:
    eta_i = intercept + x_i beta
    p_i   = sigmoid(eta_i)
    r_i   = p_i - y_i
    w_i   = p_i * (1 - p_i)
    scatter r_i and w_i into full gradient and diagonal Hessian

intercept <- intercept - g0 / h0
beta_j    <- soft_threshold(beta_j - g_j / h_j, lambda / h_j)
line search on the exact logistic L1 objective
```

This is not production Cyclops code and does not change the retained solver. The
point was to check whether loading `p_i` once per row and taking diagonal
Newton/WLS-style full-batch steps could replace many coordinate scans.

50k fold 1, variance `0.02600871`, `threads=3`:

| max outer | build sec | solve sec | final objective | final KKT | final nonzero | final gradient pass sec |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 0.128 | 1.095 | 1926.903 | 20.291 | 215 | 0.00478 |
| 240 | 0.129 | 3.706 | 1925.963 | 5.022 | 275 | 0.00450 |

200k fold 1, variance `0.01228495`, `threads=3`:

| max outer | build sec | solve sec | final objective | final KKT | final nonzero | final gradient pass sec |
|---:|---:|---:|---:|---:|---:|---:|
| 80 | 0.560 | 6.234 | 6775.639 | 53.995 | 1507 | 0.01781 |

Interpretation:

- The row-pass gradient/Hessian itself is fast, as expected.
- The diagonal/Jacobi-style update is not a viable exact solver: it requires many
  damped steps, activates far too many coefficients, and remains far from KKT.
- This is the same practical failure mode as the earlier Lipschitz/FISTA
  shortcuts, but with a more glm-like quadratic approximation: cheap full passes
  are not enough without a stronger inner solver or working set.
- Do not implement this diagonal row-pass method in production.
- If IRLS/WLS is revisited, it should use a real inner sparse-residual coordinate
  solver or a very small active-set Gram solve, followed by exact fitted-response
  KKT repair. A full-batch diagonal WLS update is not enough.

#### Indicator-column storage check

The row-pass diagnostic initially assumed all PLP columns were binary-one
indicators, but the matrix RDS has a small number of non-unit sparse values. This
raised a possible concern that Cyclops might be treating mostly-binary data as
generic sparse columns and missing the indicator fast path.

The check showed that this is not a missing optimization:

```text
200k dgCMatrix:
  all-one columns: 16,672 / 16,676
  all-one nnz:     42.398M / 42.953M = 98.7%
  non-unit nnz:     0.372M = 0.87%

50k converted Cyclops data:
  indicator columns: 14,432
  sparse columns:         4
  intercept columns:      1
```

Cyclops already preserves the PLP indicator structure, so the retained
fitted-response indicator specialization is firing for essentially all hot
columns. Further speedups need to reduce scans or improve row-state access, not
recover a missing indicator/sparse conversion.

#### Rejected native compiler flags experiment

Checked the active R toolchain:

```text
R CMD config CXX20FLAGS: -g -O2
user Makevars CXXFLAGS:  -O2 -g -fPIC
package PKG_CXXFLAGS:    -g1
```

So the normal local build is already optimized with `-O2`. A first attempt to set
`CXXFLAGS`/`CXX20FLAGS` in the environment was ignored by R's Makevars resolution,
so the experiment was rerun using a temporary `R_MAKEVARS_USER` file. Compile
commands then confirmed:

```text
-O3 -march=native -g -fPIC
```

50k exact auto-CV, `folds=3`, `threads=3`, `CYCLOPS_FOLD_LOCAL_CV=1`:

```text
normal O2 restored checks: 2.48-2.55s, variance 0.02600871, nonzero 131
O3 -march=native build:   2.61s,      variance 0.02600871, nonzero 131
```

The native build preserved correctness but was slower on the smoke benchmark, so
it was not carried to 200k. The installed package was restored to the normal `-O2`
build. Do not count custom native compiler flags as a speedup for this workload.
The likely reason is that `xBetaKnown` is false at the clone point for the
nonzero source state, so there is no cheap already-current fitted state to copy.

Also tried a warm-start `X beta` row copy after fold-local construction:

```text
CYCLOPS_FOLD_LOCAL_COPY_XBETA_WARM=1
sourceXBeta = source.getXBeta()
foldLocal.hXBeta[i] = sourceXBeta[foldLocalSourceRows[i]]
foldLocal.xBetaKnown = true
```

Fair same-build 200k comparison:

| path | elapsed sec | variance | nonzero | KKT |
|---|---:|---:|---:|---:|
| fold-local | 19.904 | 0.0122849480 | 253 | 10.475 |
| fold-local + warm `X beta` copy | 20.492 | 0.0122849480 | 253 | 10.475 |

Conclusion: both fitted-copy and warm `X beta` copy are negative/diagnostic
results. The fitted-copy path currently does not fire in auto-CV; forcing `X
beta` materialization is slower. Do not treat copy-fitted as a retained speedup.
The useful remaining fold-local question is whether the initial
constructor/setup path can avoid the `computeXBeta()` fallback without forcing a
full source materialization.

Tried a sequential prebuild of the fold-local CCD pool before the CV scheduler,
using the master CCD state as the clone source. This was worse:

```text
elapsed:                        25.146s
foldLocalBuildSeconds:           6.387s
foldLocalDataBuildSeconds:       0.813s
foldLocalSpecificsBuildSeconds:  0.000005s
foldLocalCcdSetupSeconds:        5.574s
GH seconds:                     17.129s
update seconds:                  3.441s
```

Conclusion: do not keep or reintroduce that prebuild path. It shifts/duplicates
setup work without reducing the expensive clone initialization. A useful
precomputation would need to cache reusable fold-local data/state directly, not
construct extra full CCD objects from the wrong source state.

### Compact validation fold-local experiment

The production fold-local path still scores validation folds on the full global
data object:

```text
training:   compact fold-local data
validation: global data + validation weights
```

There was already an unused `foldLocalValidationCcdPool`, so a contained
experiment was added behind:

```text
CYCLOPS_FOLD_LOCAL_VALIDATION=1
```

The experiment lazily builds a compact validation row subset for each fold, then
scores on that compact validation object. The first implementation changed the
auto-CV path because it stopped updating the global `ccdTask` to the fold's
fitted beta after validation scoring. That global CCD is still used as the
warm-start source on the next candidate. Fix: after compact validation scoring,
call:

```text
validationCcdTask->setBeta(fitCcdTask->getBetaVector())
```

This is beta-only synchronization: it preserves warm-start semantics without
computing full-data `X beta`.

After that fix, candidate/fold scores matched the baseline to numerical
precision on 50k:

```text
candidate mean max abs diff: 5.68e-13
fold score max abs diff:     7.96e-13
```

But performance was worse because validation compact clones doubled fold-local
build work:

| sample | path | elapsed sec | variance | nonzero | KKT | fold builds | build sec | data build sec | CCD setup sec | GH sec | update sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | fold-local | 4.818 | 0.02600871 | 131 | 1.543 | 3 | 1.353 | 0.194 | 1.159 | 4.201 | 0.490 |
| 50k | fold-local + compact validation | 5.032 | 0.02600871 | 131 | 1.543 | 6 | 2.021 | 0.367 | 1.654 | 4.346 | 0.495 |
| 200k | fold-local + compact validation | 21.423 | 0.0122849480 | 253 | 10.475 | 6 | 8.768 | 1.452 | 7.316 | 17.182 | 3.402 |

The comparable same-build 200k fold-local baseline before this experiment was
about `19.9s`, with three fold-local builds and about `5.5s` build time.

Conclusion: compact validation is correct after beta-only global source sync,
but negative for performance in this implementation. It mostly adds clone setup
cost, and the avoided full-data validation work is not large enough in the
current exact auto-CV path. Keep it gated as diagnostic code or remove it before
turning fold-local into a production PR.

### Skip legacy sparse-index materialization for sparse logistic folds

The split fold-local build counters showed that nearly all remaining compact
fold build cost was CCD/model initialization, not row-copy data construction or
initial `X beta` computation:

```text
200k fold-local same-build baseline:
foldLocalBuildSeconds:                 6.153s
foldLocalDataBuildSeconds:             0.814s
foldLocalCcdSetupSeconds:              5.339s
foldLocalInitSeconds:                  5.338s
foldLocalInitialComputeXBetaSeconds:   0.0002s
```

Inspection found that logistic fold-local `ModelSpecifics::initialize()` still
unconditionally called `setupSparseIndices(N)`. That constructs a legacy
per-column vector of unique patient/row indices used by the generic Cox/logistic
gradient code. The fitted-response sparse logistic kernel does not use that data:
it scans the compressed column vectors and fold-local weighted column views
directly.

A diagnostic switch was added:

```text
CYCLOPS_SPARSE_LOGISTIC_SKIP_SPARSE_INDICES=1
```

When set for logistic models without accumulation vectors, initialization leaves
`sparseIndices` empty/null instead of materializing the legacy index vectors.
This must stay tied to the sparse logistic path: if execution falls back to the
generic legacy gradient/Hessian implementation, those indices are required.

Exact auto-CV checks, `folds=3`, `threads=3`, same folds/tolerance/objective:

| sample | path | elapsed sec | variance | nonzero | KKT | build sec | data build sec | CCD setup sec | init sec | GH sec | update sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | fold-local baseline | 4.818 | 0.02600871 | 131 | 1.543 | 1.353 | 0.194 | 1.159 | - | 4.201 | 0.490 |
| 50k | fold-local + skip sparse indices | 2.688 | 0.02600871 | 131 | 1.543 | 0.194 | 0.191 | 0.003 | 0.003 | 4.214 | 0.485 |
| 200k | fold-local baseline, same build | 21.388 | 0.0122849480 | 253 | 10.475 | 6.153 | 0.814 | 5.339 | 5.338 | 17.516 | 3.459 |
| 200k | fold-local + skip sparse indices | 11.293 | 0.0122849480 | 253 | 10.475 | 0.786 | 0.774 | 0.011 | 0.010 | 17.061 | 3.348 |

This is a strong positive result. It preserves the exact auto-CV search path and
removes almost all fold-local setup overhead:

```text
50k speedup vs fold-local baseline:       1.79x
200k speedup vs same-build fold-local:    1.89x
200k fold-local init reduction:           5.338s -> 0.010s
```

The kernel counters are essentially unchanged, which confirms this is not a
different numerical solve. The speedup comes from avoiding dead legacy setup work
that was paid once per fold-local training clone.

Production follow-up: replace the environment switch with an internal condition
that only skips `setupSparseIndices()` when the selected model/kernel cannot
fall back to the generic sparse-index-based path. If fallback remains possible,
either keep materializing sparse indices or add an explicit guard that rejects
fallback when sparse indices were intentionally skipped.

That follow-up was partially implemented as lazy sparse-index materialization.
`ModelSpecifics` now tracks whether legacy sparse indices have been built. The
generic legacy entry points that directly use `sparseIndices`
(`computeGradientAndHessian`, `computeThirdDerivative`, and
`computeNumeratorForGradient`) call an `ensureSparseIndices()` guard. This means
the diagnostic skip no longer silently breaks a legacy fallback: the first legacy
use rebuilds the indices and pays the old cost at that point.

Post-guard checks:

| sample | path | elapsed sec | variance | nonzero | KKT | build sec | data build sec | CCD setup sec | init sec | GH sec | update sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | sparse-residual + skip sparse indices + lazy guard | 2.692 | 0.02600871 | 131 | 1.543 | 0.188 | 0.185 | 0.003 | 0.003 | 4.291 | 0.472 |
| 50k | legacy + skip sparse indices + lazy guard | 4.459 | 0.02600871 | 131 | 1.543 | 0.198 | 0.195 | 0.003 | 0.003 | 0 | 0 |
| 200k | sparse-residual + skip sparse indices + lazy guard | 11.180 | 0.0122849480 | 253 | 10.475 | 0.809 | 0.799 | 0.011 | 0.009 | 17.104 | 3.341 |

The legacy check is not a legacy speed claim; it verifies that the lazy guard can
recover when the sparse-index-based path is reached. The sparse-residual result
is the retained performance result: same exact search path, with fold-local
construction reduced to row-copy cost plus near-zero CCD setup.

Follow-up: the env switch was removed from the retained path by making sparse
indices lazy for logistic models by default. `ModelSpecifics` now starts logistic
fits with `sparseIndicesReady = false`; fitted-response sparse logistic paths do
not materialize the legacy sparse-index vectors, while generic legacy entry
points call `ensureSparseIndices()` before direct sparse-index use.

Default lazy checks, no `CYCLOPS_SPARSE_LOGISTIC_SKIP_SPARSE_INDICES` env var:

| sample | path | elapsed sec | variance | nonzero | KKT | build sec | data build sec | CCD setup sec | init sec | GH sec | update sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | fold-local sparse-residual default lazy | 2.723 | 0.02600871 | 131 | 1.543 | 0.192 | 0.189 | 0.003 | 0.003 | 4.282 | 0.506 |
| 50k | fold-local legacy default lazy | 4.457 | 0.02600871 | 131 | 1.543 | 0.189 | 0.186 | 0.003 | 0.003 | 0 | 0 |
| 200k | fold-local sparse-residual default lazy | 11.220 | 0.0122849480 | 253 | 10.475 | 0.749 | 0.738 | 0.010 | 0.009 | 17.146 | 3.326 |

The default-lazy sparse-residual timings match the env-gated diagnostic timings
within normal run-to-run noise. The 50k legacy run verifies that the old
sparse-index-based path can still recover via lazy materialization instead of
crashing or silently using empty sparse indices.

Current same-build comparison against the non-fold-local fitted-response path,
after default-auto update-row filtering:

| sample | path | elapsed sec | variance | nonzero | KKT | CCD epochs | GH sec | update sec | GH nnz | update nnz |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | no fold-local sparse-residual | 4.319 | 0.02747748 | 140 | 1.609 | 1176 | 7.009 | 0.841 | 8.741B | 0.948B |
| 50k | fold-local sparse-residual default lazy | 2.585 | 0.02600871 | 131 | 1.543 | 672 | 4.157 | 0.480 | 5.112B | 0.523B |
| 200k | no fold-local sparse-residual | 12.378 | 0.01154783 | 241 | 100.173 | 813 | 20.004 | 3.543 | 23.814B | 4.294B |
| 200k | fold-local sparse-residual default lazy | 11.058 | 0.0122849480 | 253 | 10.475 | 702 | 17.070 | 3.276 | 21.306B | 3.952B |

### Hot-loop profiling counter gate

The detailed gradient/Hessian and update counters were originally collected
unconditionally inside the coordinate hot loops:

```text
steady_clock start/end
gradientHessianCalls/updateCalls increments
nnzGradientHessian/nnzUpdate increments
seconds accumulation
```

That is useful for diagnostics but adds work to every coordinate call. The
per-coordinate hot-loop timing and scan counters are now gated behind:

```text
CYCLOPS_SPARSE_LOGISTIC_PROFILE=1
```

When unset, exact solver behavior is unchanged but `gradientHessianSeconds`,
`updateSeconds`, `nnzGradientHessian`, and `nnzUpdate` are not collected. Setup,
screening, KKT, fold-local build, and high-level correctness metrics remain
available.

Same-build exact auto-CV, `folds=3`, `threads=3`, no concurrent benchmarks:

| sample | env | path | elapsed sec | variance | nonzero | KKT |
|---|---|---|---:|---:|---:|---:|
| 50k | profile off | no fold-local sparse-residual | 3.799 | 0.02747748 | 140 | 1.609 |
| 50k | profile off | fold-local sparse-residual | 2.450 | 0.02600871 | 131 | 1.543 |
| 50k | profile off | fold-local sparse-residual-strong | 2.069 | 0.02600872 | 131 | 1.543 |
| 50k | profile on | fold-local sparse-residual | 2.633 | 0.02600871 | 131 | 1.543 |
| 50k | profile on | fold-local sparse-residual-strong | 2.185 | 0.02600872 | 131 | 1.543 |
| 200k | profile off | no fold-local sparse-residual | 11.879 | 0.01154783 | 241 | 100.173 |
| 200k | profile off | fold-local sparse-residual | 10.445 | 0.01228495 | 253 | 10.475 |
| 200k | profile off | fold-local sparse-residual-strong | 9.858 | 0.01223045 | 251 | 10.903 |
| 200k | profile on | fold-local sparse-residual | 10.852 | 0.01228495 | 253 | 10.475 |
| 200k | profile on | fold-local sparse-residual-strong | 9.972 | 0.01223045 | 251 | 10.903 |

Measured effect:

```text
50k fold-local sparse-residual profiling overhead: 2.633 / 2.450 = 1.07x
200k fold-local sparse-residual profiling overhead: 10.852 / 10.445 = 1.04x
```

Current unprofiled speedups versus the same-build no-fold-local fitted-response
path:

```text
50k fold-local sparse-residual:        3.799 / 2.450 = 1.55x
50k fold-local sparse-residual-strong: 3.799 / 2.069 = 1.84x
200k fold-local sparse-residual:       11.879 / 10.445 = 1.14x
200k fold-local sparse-residual-strong:11.879 /  9.858 = 1.20x
```

This is worth keeping because it is exact and removes diagnostic overhead from
normal runs. It does not meet the 50% stable speedup goal on 200k.

This is the honest current goal status:

```text
50k speedup vs current no-fold-local fitted-response:   4.319 / 2.585 = 1.67x
200k speedup vs current no-fold-local fitted-response: 12.378 / 11.058 = 1.12x
```

Against the pre-auto-update no-fold-local fitted-response measurements from the
same session, the combined retained path is:

```text
50k:  4.638 / 2.585 = 1.79x
200k: 13.267 / 11.058 = 1.20x
```

Tried removing repeated `getSparseLogisticColumnScanSize()` calls from the hot
counter path by carrying the local loop length into the counters. This was
discarded: it did not improve wall time and, more importantly, changed the
reported no-fold-local `nnzGradientHessian` count slightly. The source and
installed package were reverted to the exact counter path; only the default-auto
weighted update-row filtering was retained from this round.

The same-build 200k fold-local setup improvement is much larger:

```text
fold-local eager sparse indices: 21.388s
fold-local default lazy:         11.220s
speedup:                         1.91x
```

but that is specifically a fold-local setup-tax fix, not a 1.9x improvement over
the original no-fold-local fitted-response kernel. The larger 50k speedup also
comes with the existing weighted-view versus compact-fold search-path difference:
the selected variance changes. Keep reporting this caveat until CV fold
convergence/search-path equivalence is resolved or a fixed-candidate replay is
used for strict comparisons.

Native `perf stat` on the current 200k sparse-residual + fold-local + skip
sparse-indices path, under perf overhead:

```text
elapsed under perf:                  11.711s
variance / nonzero / KKT:            unchanged
foldLocalBuildSeconds:                0.791s
foldLocalCcdSetupSeconds:             0.0106s
foldLocalInitSeconds:                 0.0093s
gradientHessianSeconds:              17.663s
updateSeconds:                        3.549s

cpu_core cycles:                    148.8B
cpu_core instructions:              354.7B
cpu_core IPC:                         2.38
cpu_core L1D load miss rate:         15.75%
cpu_core LLC load miss rate:         58.18%
cpu_core dTLB load miss rate:         0.02%
```

Interpretation: the setup optimization removes the fold-local construction
penalty, but it does not change the remaining hot sparse gather/update kernel.
Cache behavior is improved versus earlier fold-local perf snapshots for L1D
misses, but LLC miss rate remains high. The next kernel-level opportunity is
still reducing or reorganizing repeated `fitted[row]` sparse gathers; this
specific change only fixes the fold-local setup tax.

Fresh default-lazy `perf stat` check, no sparse-index skip env var:

| path | benchmark elapsed | perf wall | variance | nonzero | KKT | core cycles | core instructions | core IPC | core L1D miss rate | core LLC miss rate | core dTLB miss rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no fold-local sparse-residual | 13.558 | 20.178 | 0.01154783 | 241 | 100.173 | 180.9B | 420.1B | 2.32 | 18.48% | 57.91% | 0.02% |
| fold-local sparse-residual default lazy | 10.988 | 17.613 | 0.01228495 | 253 | 10.475 | 149.8B | 356.0B | 2.38 | 15.74% | 62.68% | 0.02% |

The first perf row was an accidental no-fold-local run caused by omitting
`CYCLOPS_FOLD_LOCAL_CV=1`; keep it as a useful baseline, but do not label it as
fold-local. The retained fold-local/default-lazy path reduces core cycles by about
17%, core instructions by about 15%, and L1D misses by about 26% versus that
non-fold-local perf sample. LLC miss rate remains high, so the main remaining
kernel bottleneck is still sparse `fitted[row]` gather locality rather than setup.

Follow-up after gating hot-loop counters off by default:

Native `perf stat` on the current 200k `sparse-residual` + fold-local path,
`folds=3`, `threads=3`, no concurrent benchmarks:

```text
benchmark elapsed:                  10.318s
variance / nonzero / KKT:            0.01228495 / 253 / 10.475

core cycles:                       147.376B
core instructions:                 354.042B
core IPC:                            2.40
core L1D loads:                     86.768B
core L1D misses:                    13.770B
core L1D miss rate:                 15.87%
core LLC loads:                    213.342M
core LLC misses:                   129.410M
core LLC miss rate:                 60.66%
core dTLB misses:                   15.003M
core dTLB miss rate:                 0.02%
```

Native `perf record/report` on the same path, restricted to `Cyclops.so`, showed:

```text
computeSparseLogisticGradientAndHessian:          54.94%
updateSparseLogisticState:                         9.95%
CompressedDataMatrix<double>::getCompressedColumnVector: 2.45%
ModelData<double>::createRowSubsetFromBinaryWeights:     1.29%
```

Interpretation: after the setup and instrumentation fixes, almost two thirds of
sampled Cyclops time is still in the two sparse fitted-response loops. The
accessor and fold-local row-subset builder are not the main remaining costs. The
dominant operation remains sparse indexed loads from `fitted[row]` plus the
probability update writes.

Post-complement-cleanup perf check on the same retained path:

```text
benchmark elapsed:                  10.215s
variance / nonzero / KKT:            0.01228495 / 253 / 10.47502

core cycles:                       144.868B
core instructions:                 353.997B
core IPC:                            2.44
core L1D loads:                     86.734B
core L1D misses:                    13.802B
core L1D miss rate:                 15.91%
core LLC loads:                    206.487M
core LLC misses:                   122.744M
core LLC miss rate:                 59.44%
core dTLB misses:                   14.959M
core dTLB miss rate:                 0.02%
```

This is consistent with the earlier profile: the current retained kernel is
still limited by sparse fitted-response gather locality, with negligible TLB
pressure and no meaningful remaining fold-local setup cost.

Rejected all-unit fold-local view fast path:

Compact fold-local training rows have unit weights, so a test skipped
precomputed weighted column views and routed gradient/Hessian/update loops back
through the native column arrays when all weights were 1. This was negative:

```text
50k fold-local sparse-residual:        2.467s vs 2.450s prior
50k fold-local sparse-residual-strong: 2.191s vs 2.069s prior
200k fold-local sparse-residual:      10.795s vs 10.445s prior
```

Correctness was unchanged, but wall time regressed. The likely reason is that the
prebuilt fold-local weighted-view row arrays are already the compact hot-loop
representation; bypassing them reintroduces native column accessor overhead and
does not improve locality. The experiment was reverted. Keep the weighted views
on compact folds.

Rejected GH branch-hoist micro-optimization:

The current GH loops check `computeHessian` inside the per-column loops so the
same function can serve both normal CCD coordinates and gradient-only KKT scans.
Normal CCD almost always has `computeHessian = true`, so a contained test hoisted
that branch outside the inner loops and duplicated the gradient-only and
gradient-plus-Hessian loop bodies.

Results after a full preclean rebuild:

```text
50k fold-local sparse-residual:        2.516s vs previous 2.450s
50k fold-local sparse-residual-strong: 2.128s vs previous 2.069s
200k fold-local sparse-residual:      10.822s vs previous 10.445s
```

Correctness was unchanged:

```text
50k variance/nonzero/KKT:  0.02600871 / 131 / 1.542626
200k variance/nonzero/KKT: 0.01228495 / 253 / 10.47502
```

The change was reverted. The likely explanation is code-size/instruction-cache
and branch-layout cost: the branch is predictable, while the loop is dominated by
indexed `fitted[row]` loads. This is another signal that small scalar arithmetic
or branch cleanups are not the main remaining path; the next material attempt
should change data layout or reduce repeated sparse gathers.

Rejected coordinate-order experiment:

Tried an env-gated CCD traversal order for the sparse-residual kernel:

```text
CYCLOPS_SPARSE_LOGISTIC_COORDINATE_ORDER=nnz-desc
CYCLOPS_SPARSE_LOGISTIC_COORDINATE_ORDER=nnz-asc
```

The order was built once per `findMode()` call from
`getSparseLogisticColumnScanSize(j, useCrossValidation)`, so it used the same
fold-local/weighted scan lengths as the hot kernel. This changed only coordinate
visit order, not the objective, folds, tolerance, or final KKT calculation.

50k exact auto-CV, `folds=3`, `threads=3`, no concurrent benchmarks:

```text
default:  2.521s, variance 0.02600871, nonzero 131, KKT  1.542626
nnz-desc: 2.521s, variance 0.02631638, nonzero 135, KKT 10.48509
nnz-asc:  3.370s, variance 0.02653511, nonzero 142, KKT  4.673258
```

This was not worth carrying to 200k: descending order gave no speedup and moved
the auto-CV solution/KKT diagnostics; ascending order was clearly slower. The
experiment was reverted. The result suggests simple column-length ordering does
not improve the dominant `fitted[row]` locality enough to compensate for its
changed convergence path. More promising next steps need a stronger data-layout
change or fewer repeated sparse gathers, not only a different cyclic order.

Rejected x8 unroll experiment:

The current fitted-response indicator loops use four independent accumulators in
gradient/Hessian and update four fitted probabilities per iteration in the state
update loop. Tried increasing the unit-weight indicator gradient/Hessian path and
indicator update path to x8 unrolling.

50k exact auto-CV, `folds=3`, `threads=3`, same source except unroll width:

```text
restored x4: 2.484s, variance 0.02600871, nonzero 131, KKT 1.542626
x8 unroll:   2.615s, variance 0.02600871, nonzero 131, KKT 1.542626
```

The result is exact but slower, so the x8 change was reverted. This is consistent
with the other micro-kernel negatives: the loop is not limited by scalar loop
overhead, and larger loop bodies/register pressure can hurt while the real cost
remains sparse indexed `fitted[row]` loads and probability writes.

Rejected run-length indicator-column representation:

Added a benchmark-only diagnostic:

```text
extras/benchmarks/diagnose_indicator_runs.R
```

It remaps each training fold to compact local row ids, then measures exact
consecutive row-id runs inside all-one indicator columns. A possible kernel would
replace individual row-index loads with:

```text
for each contiguous segment [start, end):
    for row in start:end:
        p = fitted[row]
        accumulate gradient/Hessian
```

This only helps if a large share of hot-column nnz are in nontrivial exact runs.

50k fold-local diagnostic:

```text
hot-column nnz in runs length >= 2: 47.6-47.8%
hot-column nnz in runs length >= 4: 10.6-11.0%
hot-column nnz in runs length >= 8:  1.18-1.24%
hot average run length:              1.40-1.41
```

200k fold-local diagnostic:

```text
hot-column nnz in runs length >= 2: 38.3%
hot-column nnz in runs length >= 4:  7.46-7.54%
hot-column nnz in runs length >= 8:  0.73-0.75%
hot average run length:              1.29-1.30
```

Decision: do not implement a run-length segment kernel. Exact contiguous runs are
too short and too rare. The earlier locality improvement from compact folds comes
from smaller row deltas and better cache-line reuse, not from long contiguous
ranges that can be scanned as dense slices.

Rejected block-coded 16-bit row-offset representation:

Added a benchmark-only diagnostic:

```text
extras/benchmarks/diagnose_block16_row_kernel.R
```

It compares two exact indicator-column gradient/Hessian microkernels on compact
training folds:

```text
current:
    int row = rows[k]
    p = fitted[row]

block16:
    int base = blockBase[b]
    uint16 offset = offsets[k]
    p = fitted[base + offset]
```

The block16 format groups row ids into 65,536-row blocks, stores one 32-bit base
per block, and stores 16-bit offsets. It keeps the same arithmetic and only tests
whether row-index bandwidth is a large enough part of the hot loop.

50k fold 1:

```text
indicator nnz:       7.067M
block count:         13,749
row-index bytes:     28.3MB -> 14.3MB
int median sec:      0.002085
block16 median sec:  0.001712
speedup:             1.22x
checksum diff:       0
```

200k folds:

```text
fold 1: row-index bytes 112.8MB -> 56.8MB, speedup 1.039x
fold 2: row-index bytes 113.3MB -> 57.0MB, speedup 1.035x
fold 3: row-index bytes 113.1MB -> 56.9MB, speedup 1.015x
```

Decision: do not implement block16 row storage in the production kernel now. It
shows row-index bandwidth matters on 50k, but the 200k target only gets a
1-4% isolated GH microkernel gain. That is too small to move exact auto-CV
materially once update loops, build cost, code complexity, and non-indicator
columns are included. The dominant 200k cost is still the sparse `fitted[row]`
load itself, not row-index byte width.

### CV scoring / beta-sync timing diagnostic

Added split counters for validation scoring overhead in the CV driver:

```text
sparseLogistic.cvBetaSyncSeconds
sparseLogistic.cvScoringSeconds
```

These counters time the post-fit validation path separately from the fold-local
training solve:

```text
beta sync:
    copy/update beta from compact training CCD into the validation/full-data CCD

validation scoring:
    compute predictive log-likelihood for the held-out fold
```

Benchmarks used exact auto-CV, `folds=3`, `threads=3`,
`CYCLOPS_FOLD_LOCAL_CV=1`, and `CYCLOPS_SPARSE_LOGISTIC_PROFILE=1`.

| sample | elapsed sec | variance | nonzero | KKT | GH sec | update sec | fold-local build sec | beta sync sec | validation scoring sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | 2.482 | 0.02600871 | 131 | 1.543 | 4.015 | 0.477 | 0.191 | 0.034 | 0.0046 |
| 200k | 10.481 | 0.01228495 | 253 | 10.475 | 16.288 | 3.251 | 0.730 | 0.307 | 0.0247 |

Notes:

- `GH sec` and `update sec` are summed across fold threads, so they can exceed
  wall time. They are still useful for comparing where threaded work is spent.
- Validation scoring itself is negligible: about 0.2% of wall time on both 50k
  and 200k.
- Beta sync is small but visible: about 1.4% of wall time on 50k and 2.9% on
  200k.
- Fold-local build is larger than beta sync/scoring, but still only about 7% of
  wall time on these runs.

Decision: do not optimize validation scoring next. Even removing all validation
scoring time would not materially move the exact CV benchmark. Removing beta
sync entirely would also be too small to approach the 50% speedup goal. The next
useful work should continue to target the sparse training hot path and
fold-local layout/build costs, especially repeated `fitted[row]` gathers and
probability writes.

Same-build 200k weighted-view versus fold-local comparison after adding the
scoring counters:

| path | elapsed sec | variance | nonzero | KKT | GH sec | update sec | GH nnz | update nnz | CCD epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weighted-view | 11.560 | 0.01154783 | 241 | 100.173 | 19.279 | 3.498 | 23.814B | 4.294B | 813 |
| fold-local | 10.481 | 0.01228495 | 253 | 10.475 | 16.288 | 3.251 | 21.306B | 3.952B | 702 |

Current same-build fold-local speedup:

```text
wall:       11.560 / 10.481 = 1.10x
GH sec:     19.279 / 16.288 = 1.18x
update sec:  3.498 /  3.251 = 1.08x
```

This is a stable positive layout result but still far below the 50% target on
200k. The current fold-local setup cost is no longer the earlier 5-6 second
problem; with lazy sparse-index setup it is about 0.7-0.8 seconds on 200k.

Candidate/fold traces on the same current binary show the remaining production
caveat clearly. Both paths evaluate the same three auto-CV candidates, but
fold-level scores differ from the first candidate:

| step | fold | variance | weighted pred LL | fold-local pred LL | delta | weighted KKT | fold-local KKT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.01342480 | -2746.232 | -2758.923 | -12.690 | 48.981 | 3.579 |
| 0 | 2 | 0.01342480 | -2694.782 | -2705.535 | -10.753 | 75.814 | 6.470 |
| 0 | 3 | 0.01342480 | -2423.349 | -2434.261 | -10.912 | 59.032 | 1.738 |
| 1 | 1 | 0.13424800 | -2846.446 | -2851.027 | -4.581 | 26.407 | 4.910 |
| 1 | 2 | 0.13424800 | -2737.200 | -2740.080 | -2.880 | 15.640 | 4.817 |
| 1 | 3 | 0.13424800 | -2484.388 | -2487.918 | -3.530 | 26.611 | 3.910 |
| 2 | 1 | 0.00134248 | -2792.491 | -2792.689 | -0.199 | 5.551 | 2.864 |
| 2 | 2 | 0.00134248 | -2777.092 | -2776.490 | 0.602 | 4.358 | 3.907 |
| 2 | 3 | 0.00134248 | -2451.329 | -2484.042 | -32.713 | 212.434 | 8.630 |

The candidate means differ by up to 11.45 log-likelihood units and the selected
variance shifts:

```text
weighted-view selected variance: 0.01154783
fold-local selected variance:    0.01228495
```

Interpretation: this is not a validation-scoring implementation problem. The
fold-local path scores on the full validation row space, and compact validation
previously matched the full-data scoring path to numerical precision after
beta-only warm-start synchronization. The difference is the training fold solve
itself: compact folds reach substantially smaller full-KKT residuals under the
same default gradient convergence criterion, so their fitted betas and validation
scores differ. That can be a better fold solution, but it means fold-local is not
yet a pure drop-in replacement for the weighted-view CV path.

Next decision: to make fold-local production-ready, either:

1. define CV fold convergence in a way that is representation-stable enough that
   weighted-view and compact-fold training solves score the same candidate path;
   or
2. explicitly accept fold-local as a changed-but-more-converged CV implementation
   and validate it against stricter objective/KKT criteria rather than requiring
   identical weighted-view search paths.

For the speed goal, option 1 is cleaner but may add work. Option 2 may be
reasonable if the current weighted-view default gradient convergence is shown to
stop with large subgradient/KKT residuals, but that is a behavioral decision for
Cyclops maintainers rather than only a kernel optimization.

#### Convergence-criterion diagnostic

The fold-local/weighted-view divergence led to a small benchmark-wrapper change:

```text
--convergence-type gradient|lange|mittal|zhang
```

This is only a benchmark option. It does not change Cyclops defaults. It was used
to test whether using the log-posterior (`lange`) convergence criterion would be
more representation-stable or faster than the default `gradient` criterion.

Exact auto-CV, `folds=3`, `threads=3`, `tolerance=1e-6`:

| sample | path | convergence | elapsed sec | variance | nonzero | KKT | CCD epochs |
|---|---|---|---:|---:|---:|---:|---:|
| 50k | weighted-view | lange | 2.274 | 0.02595127 | 134 | 9.238 | 758 |
| 50k | fold-local | lange | 1.522 | 0.02588639 | 133 | 8.893 | 450 |
| 200k | weighted-view | lange | 21.540 | 0.01254762 | 254 | 24.706 | 1798 |
| 200k | fold-local | lange | 10.374 | 0.01222766 | 250 | 24.187 | 733 |

For comparison, the current 200k default-gradient same-build results were:

```text
weighted-view gradient: 11.560s, variance 0.01154783, nonzero 241, KKT 100.173
fold-local gradient:    10.481s, variance 0.01228495, nonzero 253, KKT 10.475
```

Interpretation:

- `lange` convergence is not a useful speed path on the main 200k target.
  Weighted-view becomes much slower, and fold-local is essentially unchanged in
  wall time while its KKT check worsens versus default-gradient fold-local.
- The 50k fold-local `lange` run is faster, but that improvement does not scale
  to 200k and should not drive production decisions.
- The diagnostic reinforces the core issue: Cyclops' convergence criterion
  materially changes the auto-CV path and KKT residuals. Solving representation
  equivalence by switching convergence criteria is not a clear performance win.

Follow-up: stricter default-gradient convergence does not turn fold-local into a
large speedup either. Exact auto-CV, `folds=3`, `threads=3`, `tolerance=1e-8`,
`maxIterations=5000`:

| sample | path | elapsed sec | variance | nonzero | KKT | objective |
|---|---|---:|---:|---:|---:|---:|
| 50k | weighted-view | 6.852 | 0.02648656 | 138 | 0.0152 | -2732.219 |
| 50k | fold-local | 6.809 | 0.02648010 | 138 | 0.0152 | -2732.238 |
| 200k | weighted-view | 42.680 | 0.01278032 | 259 | 0.3274 | -10124.315 |
| 200k | fold-local | 40.774 | 0.01243368 | 258 | 0.0411 | -10128.586 |

Speedups:

```text
50k strict gradient fold-local vs weighted-view:   1.01x
200k strict gradient fold-local vs weighted-view:  1.05x
```

Interpretation: tightening the existing gradient convergence criterion greatly
increases runtime and reduces KKT residuals, but it does not make compact
fold-local training materially faster than weighted-view training on the 200k
target. Fold-local still reaches a substantially lower KKT residual at 200k, but
the wall-time gain is only about 5%. This rules out "make convergence stricter"
as the missing route to the 50% speedup goal.

#### Unconditional prefetch retry

Retested a direct one-cache-line-ahead software prefetch in the fitted-response
indicator hot loops:

```text
computeSparseLogisticGradientAndHessian:
    __builtin_prefetch(fitted + rows[row + 16], 0, 1)

updateSparseLogisticState:
    __builtin_prefetch(fitted + rows[row + 16], 1, 1)
```

This was a straight-line default patch, not the earlier env-gated version. It was
tested with exact auto-CV, `folds=3`, `threads=3`,
`CYCLOPS_FOLD_LOCAL_CV=1`.

| sample | path | elapsed sec | variance | nonzero | KKT |
|---|---|---:|---:|---:|---:|
| 50k | fold-local + prefetch distance 16 | 2.472 | 0.02600871 | 131 | 1.543 |
| 200k | fold-local + prefetch distance 16 | 10.715 | 0.01228495 | 253 | 10.475 |
| 50k | fold-local after removing prefetch | 2.284 | 0.02600871 | 131 | 1.543 |
| 200k | fold-local after removing prefetch | 10.209 | 0.01228495 | 253 | 10.475 |

Decision: remove the unconditional prefetch patch. It preserved correctness but
hurt wall time on the 200k target and was not a stable improvement on 50k. The
`ModelSpecifics.hpp` hot loops no longer contain `__builtin_prefetch` calls.

#### Current cleaned strong-rule combination check

After removing the unconditional prefetch patch, reran the current fold-local
baseline against historical-reference strong rules in the same benchmark process.
Exact auto-CV, `folds=3`, `threads=3`, `CYCLOPS_FOLD_LOCAL_CV=1`.

| sample | kernel | elapsed sec | variance | nonzero | KKT | objective diff vs residual | speedup vs residual |
|---|---|---:|---:|---:|---:|---:|---:|
| 50k | sparse-residual | 2.304 | 0.02600871 | 131 | 1.543 | 0 | 1.00x |
| 50k | sparse-residual-strong | 1.999 | 0.02600872 | 131 | 1.543 | 2.24e-05 | 1.15x |
| 200k | sparse-residual | 10.359 | 0.01228495 | 253 | 10.475 | 0 | 1.00x |
| 200k | sparse-residual-strong | 9.761 | 0.01223045 | 251 | 10.903 | -0.696 | 1.06x |

Decision: historical strong rules remain a small positive add-on, especially on
50k, but they do not materially change the 200k goal. The current 200k speedup is
only about 6% in a same-process comparison, so the next attempt still needs to
target the sparse fitted-response gather/update kernel itself.

Fresh cleaned 200k perf sample, same exact fold-local residual path:

```text
elapsed: 10.214s
core cycles: 145.6B
core instructions: 353.9B
core IPC: 2.43
core L1D loads: 86.7B
core L1D load misses: 13.8B  (15.9%)
core LLC loads: 209.8M
core LLC load misses: 123.7M (59.0%)
core dTLB load misses: 15.0M (0.02%)
```

Core-cycle sampled profile:

```text
55.4% computeSparseLogisticGradientAndHessian
 9.6% updateSparseLogisticState
 2.0% CompressedDataMatrix::getCompressedColumnVector
 1.3% ModelData::createRowSubsetFromBinaryWeights
 1.2% __ieee754_exp_fma
```

The core annotate view shows the retained unit-weight indicator loop is the
expected scalar gather/reduce shape:

```asm
movslq  (%rax), %rdx          ; row id
movsd   (%r15,%rdx,8), %xmm3 ; fitted[row]
addsd   %xmm3, ...
subsd/mulsd/addsd            ; p * (1 - p)
jne     loop
```

The hot instructions are row-id/fitted gathers and scalar Hessian multiplies, not
`exp` or validation/setup work. This supports the current decision to stop local
prefetch/branch/unroll tweaks and only pursue changes that reduce repeated
coordinate scans or alter the inner solver/data layout more substantially.

#### Blitz-style restricted working-set diagnostic

Added an env-gated diagnostic path for `sparse-residual-strong`:

```text
CYCLOPS_STRONG_WORKING_SET=1
CYCLOPS_WORKING_SET_INITIAL_SIZE=<n>
CYCLOPS_WORKING_SET_REPAIR_SIZE=<n>
CYCLOPS_WORKING_SET_MAX_REPAIRS=<n>
```

This does not affect the default strong-rule path. The diagnostic starts from
unpenalized/current-nonzero/reference-nonzero coordinates plus top current KKT
violators, solves the restricted problem, then runs full KKT repair and adds top
violators. Exactness still requires a final full-KKT pass; fallback uses the
normal sparse-residual solve.

50k exact auto-CV, `folds=3`, `threads=3`, `CYCLOPS_FOLD_LOCAL_CV=1`:

| setting | kernel | elapsed sec | variance | nonzero | KKT | CCD epochs | active-only epochs | full KKT scans | GH nnz | KKT nnz | objective diff vs residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | sparse-residual | 2.314 | 0.02600871 | 131 | 1.543 | 672 | 0 | 0 | 0 | 0 | 0 |
| baseline | sparse-residual-strong | 1.960 | 0.02600872 | 131 | 1.543 | 667 | 206 | 3 | 17.5M | 17.5M | 2.24e-05 |
| WS 512/512/8 | sparse-residual-strong | 3.647 | 0.02647569 | 137 | 0.022 | n/a | n/a | n/a | n/a | n/a | n/a |
| WS 2048/2048/8 | sparse-residual-strong | 3.752 | 0.02647633 | 137 | 0.017 | 1903 | 1903 | 20 | 72.3M | 41.0M | 1.379 |

Decision: reject this diagnostic in its current form. It creates much smaller
working sets, but the restricted problems are solved and repaired so many times
that total work increases. In the 2048 case, the active-set size mean dropped
from about `10210` to about `918`, but gradient/Hessian work rose from `17.5M`
to `72.3M` nnz and full KKT scans rose from `3` to `20`. It also changes the
auto-search result on 50k, so it is not a production candidate.

Interpretation: a Blitz-like growth wrapper around the current exact CCD is not
enough. To make small working sets useful, the restricted solve must avoid
over-solving before repair, or the inner algorithm must change more
substantially. Given the previous active-set/strong-rule results, the next
higher-probability work remains data-layout/solver-kernel changes rather than
more top-K working-set thresholding.

Follow-up tried that staged-repair hypothesis directly behind a temporary
diagnostic:

```text
CYCLOPS_STRONG_STAGED_REPAIR_EPSILON=1e-3
```

The diagnostic first solved a restricted strong set loosely to find inactive KKT
violators, then re-solved to the normal exact final tolerance and reran full
inactive KKT before accepting. The final acceptance criterion was not loosened.

50k exact auto-CV, `folds=3`, `threads=3`, `CYCLOPS_FOLD_LOCAL_CV=1`,
`CYCLOPS_SPARSE_LOGISTIC_PROFILE=1`:

| path | elapsed sec | variance | nonzero | KKT | CCD epochs | active-only epochs | full KKT scans | GH nnz | update nnz | KKT nnz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| strong baseline | 2.320 | 0.02600872 | 131 | 1.543 | 667 | 206 | 3 | 3.967B | 0.522B | 17.5M |
| staged repair 1e-3 | 2.384 | 0.02600868 | 131 | 1.543 | 668 | 207 | 8 | 3.999B | 0.522B | 46.7M |

The staged pass made the mean active set smaller (`10210` to `8566`) but added
violator cycles and KKT scans, increased gradient/Hessian work, and slightly
changed the selected variance. The temporary diagnostic was reverted without
running 200k.

Decision: reject staged restricted repair for the current strong-rule wrapper.
The failure mode is not just over-solving the first restricted set; the extra
repair loop itself adds enough work and path instability that it is not a
promising route to the 200k speed target.

#### Rejected sparse-state ready guard fast path

Tried a tiny hot-loop guard change in `ModelSpecifics.hpp`:

```cpp
if (!sparseLogisticStateReady) {
    refreshSparseLogisticState(useWeights);
}
```

instead of calling `refreshSparseLogisticState(useWeights)` directly from
`computeSparseLogisticGradientAndHessian()` and `updateSparseLogisticState()`.
The intended effect was only to avoid a repeated function call/early-return in
the coordinate hot path. The semantic initialization point was unchanged:
`findMode()` already refreshes fitted-response state before entering the sparse
logistic CCD path, and updates maintain that state exactly.

50k exact auto-CV, `folds=3`, `threads=3`, `CYCLOPS_FOLD_LOCAL_CV=1`:

| path | elapsed sec | variance | nonzero | KKT |
|---|---:|---:|---:|---:|
| sparse-residual | 2.323 | 0.02600871 | 131 | 1.543 |
| sparse-residual-strong | 2.031 | 0.02600872 | 131 | 1.543 |

This is neutral to slightly worse than the immediately preceding same-build
baseline (`2.314s` and `1.960s`). The change was reverted without running 200k.

Decision: do not pursue guard/function-call cleanup around
`refreshSparseLogisticState()`. It does not affect the real bottleneck:
row-indexed `fitted[row]` loads and probability update writes.

#### Rejected fitted-response KKT swindle diagnostic

Inspected Cyclops' existing `useKKTSwindle` active-set wrapper. The legacy path
solves restricted active sets through the templated `findMode(begin,end,...)`
overload, but that overload hard-wires `LogisticKernelType::LEGACY`, and inactive
KKT scoring calls the legacy numerator/gradient/Hessian path. A small diagnostic
patch made both pieces kernel-aware behind an env gate, and the benchmark wrapper
temporarily exposed `--use-kkt-swindle`.

50k exact auto-CV, `folds=3`, `threads=3`, `CYCLOPS_FOLD_LOCAL_CV=1`:

| path | elapsed sec | variance | nonzero | KKT | sparse lambda values | CCD epochs |
|---|---:|---:|---:|---:|---:|---:|
| fold-local sparse-residual baseline | 2.329 | 0.02600871 | 131 | 1.543 | 10 | 672 |
| sparse-residual + sparse KKT swindle | 2.358 | 0.02646428 | 137 | 0.146 | 33 | 1223 |

Decision: reject and revert without running 200k. The sparse KKT-swindle version
was slightly slower on 50k, changed the auto-CV selected variance, and more than
doubled lambda/epoch counters. This reinforces the earlier top-K working-set
result: wrapping the current exact CCD kernel in another restricted active-set
repair loop tends to over-solve restricted problems and increase total work.

#### AVX2 gather diagnostic for indicator gradient/Hessian

Added an AVX2 implementation for the unit-weight indicator gradient/Hessian
loop. It was initially env-gated:

```text
CYCLOPS_AVX2_GATHER_GH=1
```

The default scalar x4 loop remains unchanged unless the env var is set and the
runtime CPU reports AVX2 support. The patch targets only the dominant
unit-weight indicator GH loop:

```cpp
rowIndex = load 4 int row ids
p        = _mm256_i32gather_pd(fitted, rowIndex, 8)
g       += sum(p)
h       += sum(p * (1 - p))
```

Exact auto-CV, `folds=3`, `threads=3`, `CYCLOPS_FOLD_LOCAL_CV=1`,
`tolerance=1e-6`:

| sample | env | kernel | elapsed sec | variance | nonzero | KKT |
|---|---|---|---:|---:|---:|---:|
| 50k | off | sparse-residual | 2.340 | 0.02600871 | 131 | 1.543 |
| 50k | AVX2 | sparse-residual | 2.279 | 0.02600871 | 131 | 1.543 |
| 200k | off | sparse-residual | 10.360 | 0.01228495 | 253 | 10.475 |
| 200k | AVX2 | sparse-residual | 9.941 | 0.01228495 | 253 | 10.475 |
| 50k | AVX2 | sparse-residual-strong | 1.964 | 0.02600872 | 131 | 1.543 |
| 200k | AVX2 | sparse-residual-strong | 9.650 | 0.01223045 | 251 | 10.903 |

With profiling counters enabled on 200k:

| env | elapsed sec | GH sec | update sec | GH nnz | update nnz | epochs |
|---|---:|---:|---:|---:|---:|---:|
| off | 10.622 | 16.451 | 3.300 | 21.306B | 3.952B | 702 |
| AVX2 | 10.501 | 16.038 | 3.220 | 21.306B | 3.952B | 702 |

Assembly check of the installed shared object confirms the intended gather
instruction is present:

```asm
vmovdqu    (%rdi,%rax,4),%xmm7
vgatherdpd %ymm6,(%rsi,%xmm7,8),%ymm0
vaddpd     %ymm0,%ymm1,%ymm1
vsubpd/vmulpd/vaddpd              ; p * (1 - p)
```

Interpretation: manual AVX2 gather is a real but small improvement. It reduces
the profiled GH loop by about `2.5%` and unprofiled 200k wall time by about
`4%` in this run. This is nowhere near the remaining goal by itself, and it
confirms that the bottleneck is memory-latency/irregular-gather dominated rather
than scalar arithmetic throughput dominated. Keep this as an optional
env-gated diagnostic/possible add-on, but do not treat it as the main next
architecture.

Follow-up rejected an x8/two-gather variant intended to expose more
memory-level parallelism:

```text
50k  AVX2 x4: 2.279s
50k  AVX2 x8: 2.306s
200k AVX2 x4: 9.941s
200k AVX2 x8: 10.264s
```

The x8 helper preserved the exact same variance/nonzero/KKT results, but was
slower than the simpler x4 gather helper on both benchmark sizes. The source was
restored to the x4 AVX2 helper.

Follow-up retained the x4 AVX2 helper as default-on when runtime CPU detection
reports AVX2 support. The environment variable is now an escape hatch:

```text
CYCLOPS_AVX2_GATHER_GH=0
```

disables the AVX2 gather path and forces the scalar loop. On non-x86 or CPUs
without AVX2, the scalar loop remains the fallback.

Same-build A/B after switching to default-on runtime dispatch, exact auto-CV,
`folds=3`, `threads=3`, `CYCLOPS_FOLD_LOCAL_CV=1`:

| sample | path | elapsed sec | variance | nonzero | KKT | speedup |
|---|---|---:|---:|---:|---:|---:|
| 50k | forced scalar, `CYCLOPS_AVX2_GATHER_GH=0` | 2.506 | 0.02600871 | 131 | 1.543 | 1.00x |
| 50k | default AVX2 | 2.396 | 0.02600871 | 131 | 1.543 | 1.046x |
| 200k | forced scalar, `CYCLOPS_AVX2_GATHER_GH=0` | 10.814 | 0.01228495 | 253 | 10.475 | 1.00x |
| 200k | default AVX2 | 10.610 | 0.01228495 | 253 | 10.475 | 1.019x |

Profiled 200k pair:

| path | elapsed sec | GH sec | update sec | GH nnz | update nnz |
|---|---:|---:|---:|---:|---:|
| forced scalar | 11.261 | 16.968 | 3.302 | 21.306B | 3.952B |
| default AVX2 | 10.784 | 16.089 | 3.315 | 21.306B | 3.952B |

This gives:

```text
200k profiled wall speedup: 11.261 / 10.784 = 1.044x
200k profiled GH speedup:   16.968 / 16.089 = 1.055x
```

Decision: keep default-on AVX2 runtime dispatch as a small, exact hot-loop
optimization. It preserves the selected variance, nonzero count, KKT diagnostic,
and scanned nnz. It does not materially change the remaining 50% goal; it is a
low-risk add-on while larger algorithm/layout work remains unresolved.

Current-build recheck after the later fold-local/lazy changes, 200k exact
auto-CV, `folds=3`, `threads=3`, no trace/profiling:

| path | elapsed sec | variance | nonzero | KKT |
|---|---:|---:|---:|---:|
| forced scalar, `CYCLOPS_AVX2_GATHER_GH=0` | 10.969 | 0.01228495 | 253 | 10.475 |
| default AVX2 | 10.717 | 0.01228495 | 253 | 10.475 |

The current default AVX2 speedup is `10.969 / 10.717 = 1.024x`, consistent with
the earlier diagnostic. AVX2 gather is worth keeping, but it is not a path to the
remaining 50% goal by itself.

#### Rejected no-op sparse update guard

Tried an exact opt-in guard for sparse logistic state updates:

```text
CYCLOPS_SKIP_NOOP_SPARSE_UPDATES=1
```

The idea was to skip an indicator/intercept update when `exp(delta)` rounded
exactly to `1.0`, because the fitted-response update would then leave every
probability bitwise unchanged. This was exact, but it almost never triggered.

200k exact auto-CV, `folds=3`, `threads=3`, `CYCLOPS_FOLD_LOCAL_CV=1`,
`CYCLOPS_SPARSE_LOGISTIC_PROFILE=1`:

| env | elapsed sec | variance | nonzero | KKT | GH sec | update sec | GH nnz | update nnz | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 11.274 | 0.01228495 | 253 | 10.475 | 17.018 | 3.330 | 21.306B | 3.952238501B | 702 |
| no-op guard | 11.112 | 0.01228495 | 253 | 10.475 | 17.043 | 3.400 | 21.306B | 3.952007513B | 702 |

The guard skipped only about `230,988` update nonzeros out of `3.952B`
profiled update nonzeros, roughly `0.006%`. Profiled update time increased
slightly, so the wall-time difference is within noise rather than evidence of a
useful speedup.

Decision: reject and revert. Near-zero coordinate updates are already handled
well enough by the existing `delta == 0` skip; testing for an exactly neutral
probability multiplier does not remove meaningful work.

#### Rejected gradient-first zero-coordinate skip

Tried an exact env-gated coordinate-update diagnostic:

```text
CYCLOPS_GRADIENT_FIRST_ZERO_SKIP=1
```

For zero coefficients under a KKT-compatible L1 boundary, the diagnostic computed
the gradient only first. If `abs(g_j) <= lambda_j`, the coordinate update is
provably zero and the Hessian is unnecessary. If the coordinate violated the
boundary, the normal gradient+Hessian path ran afterward. This preserved the
same objective, folds, tolerance, and final KKT checks.

50k exact auto-CV, `folds=3`, `threads=3`, `CYCLOPS_FOLD_LOCAL_CV=1`,
`CYCLOPS_SPARSE_LOGISTIC_PROFILE=1`:

| env | elapsed sec | variance | nonzero | KKT | GH calls | GH sec | GH nnz | update sec | update nnz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 2.762 | 0.02600871 | 131 | 1.543 | 9.702M | 4.324 | 5.112B | 0.502 | 0.523B |
| gradient-first | 2.704 | 0.02600871 | 131 | 1.543 | 9.705M | 4.390 | 5.128B | 0.471 | 0.523B |

The tiny wall-time gain is not supported by the counters: gradient/Hessian calls,
scanned nnz, and GH time all increased. The update-time decrease is too small and
not a direct result of reduced update nnz. The likely explanation is noise plus
occasional double scans for violating zero coordinates.

Decision: reject and revert without running 200k. This does not reduce the
dominant sparse `fitted[row]` gather work.

#### Rejected GH-fitted cache for immediate update

Tried an exact env-gated handoff between the coordinate gradient/Hessian pass and
the immediately following sparse fitted-response update:

```text
CYCLOPS_CACHE_GH_FITTED_FOR_UPDATE=1
```

For currently nonzero unit-weight indicator coordinates, the diagnostic cached
the fitted probabilities loaded during `computeSparseLogisticGradientAndHessian`.
If the same coordinate then moved, `updateSparseLogisticState` used that
sequential cache as the old probability value instead of reloading
`fitted[row]` through the irregular gather path. This preserves the same CCD
algorithm and exact fitted-response update; it only changes where the update
reads the old probability from.

50k exact auto-CV, `folds=3`, `threads=3`, `CYCLOPS_FOLD_LOCAL_CV=1`,
`CYCLOPS_SPARSE_LOGISTIC_PROFILE=1`:

| env | elapsed sec | variance | nonzero | KKT | GH calls | GH sec | GH nnz | update sec | update nnz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 2.739 | 0.02600871 | 131 | 1.543 | 9.702M | 4.227 | 5.112B | 0.497 | 0.523B |
| cache | 2.733 | 0.02600871 | 131 | 1.543 | 9.702M | 4.219 | 5.112B | 0.512 | 0.523B |

The tiny wall-time difference is noise: update time increased by about `2.9%`,
while gradient/Hessian work and scanned nnz were unchanged. The likely reason is
that the cache adds extra sequential stores during the GH pass and replaces
irregular update loads with sequential cache loads, but update still must scatter
writes to `fitted[row]`; the saved gather is not enough to offset the added
traffic/bookkeeping.

Decision: reject and revert without running 200k. A fused/cached update handoff
does not materially reduce the dominant cost. Future work should avoid
per-coordinate scratch buffers unless it also reduces full-column scan count or
changes the solver/data layout more substantially.

#### Current-best native perf snapshot

Reran native profiling on the retained current-best path:

```text
CYCLOPS_FOLD_LOCAL_CV=1
kernel=sparse-residual-strong
sample=200k
folds=3
threads=3
```

The exact auto-CV benchmark reported:

| kernel | elapsed sec | variance | nonzero | KKT |
|---|---:|---:|---:|---:|
| sparse-residual-strong | 9.818 | 0.01223045 | 251 | 10.903 |

`perf stat` on the same run:

```text
cpu_core cycles:                    136.1B
cpu_core instructions:              232.8B
cpu_core IPC:                         1.71
cpu_core L1D loads:                  52.9B
cpu_core L1D load misses:            12.6B  (23.9%)
cpu_core LLC loads:                 277.4M
cpu_core LLC load misses:           179.6M  (64.7%)
cpu_core branch misses:             129.1M  (0.34%)
```

`perf record -e cpu_core/cycles/P` showed:

| symbol | core-cycle share |
|---|---:|
| `sparseLogisticAvx2IndicatorGradientHessianDouble` | 50.0% |
| `updateSparseLogisticState` | 11.2% |
| R `HashLookup` and GC | ~10.3% |
| `getCompressedColumnVector` | 1.9% |
| fold-local row subset build | 1.6% |
| `exp`/`__ieee754_exp_fma` | ~1.9% |

Interpretation: after fold-local/lazy setup and default AVX2, the remaining
hot path is still the repeated indicator-column `fitted[row]` gather in
gradient/Hessian. Branching and setup are not the main issue. LLC miss rate
remains high, but the useful target is reducing row-gather count or changing the
solver/layout so each row-state load is reused more often.

#### Rejected scan-size coordinate ordering

Tried an env-gated diagnostic coordinate order for the sparse logistic CCD loop:

```text
CYCLOPS_SPARSE_LOGISTIC_COORDINATE_ORDER=nnz-asc
CYCLOPS_SPARSE_LOGISTIC_COORDINATE_ORDER=nnz-desc
```

The implementation sorted the full CCD sweep by
`getSparseLogisticColumnScanSize(index, useCrossValidation)` and otherwise kept
the same coordinate updates and exact fitted-response state update. The intent
was to test whether grouping rare or common columns improves row-state locality
or convergence enough to matter.

200k exact auto-CV, `folds=3`, `threads=3`, `CYCLOPS_FOLD_LOCAL_CV=1`:

| order | kernel | elapsed sec | variance | nonzero | KKT |
|---|---|---:|---:|---:|---:|
| default | sparse-residual | 10.409 | 0.01228495 | 253 | 10.475 |
| default | sparse-residual-strong | 9.999 | 0.01223045 | 251 | 10.903 |
| nnz-asc | sparse-residual | 9.704 | 0.01213979 | 263 | 44.168 |
| nnz-asc | sparse-residual-strong | 7.764 | 0.01214445 | 286 | 38.672 |
| nnz-desc | sparse-residual | 10.911 | 0.01217389 | 249 | 27.815 |
| nnz-desc | sparse-residual-strong | 10.322 | 0.01228499 | 253 | 26.941 |

The ascending order looked fast under the default gradient stop, but it changed
the selected variance/nonzero count and worsened the KKT diagnostic. A stricter
follow-up rejected it:

```text
CYCLOPS_SPARSE_LOGISTIC_COORDINATE_ORDER=nnz-asc
tolerance=1e-8
maxIterations=5000
```

| kernel | elapsed sec | variance | nonzero | KKT |
|---|---:|---:|---:|---:|
| sparse-residual | 49.255 | 0.01236084 | 259 | 2.456 |
| sparse-residual-strong | 48.748 | 0.01236074 | 258 | 2.456 |

This is slower than the earlier strict fold-local baseline and does not preserve
the default path. Decision: reject and revert. Coordinate ordering is another
path-sensitivity knob, not a robust route to the 50% speedup goal under exact
CV semantics.

#### Strong-rule auto-search trace, default 10x step

Reran the current retained 200k exact auto-CV path with candidate and fold
traces:

```text
CYCLOPS_FOLD_LOCAL_CV=1
CYCLOPS_CV_CANDIDATE_TRACE=/tmp/cyclops-strong-200k-candidate-trace.csv
CYCLOPS_CV_FOLD_TRACE=/tmp/cyclops-strong-200k-fold-trace.csv
folds=3
threads=3
kernel=sparse-residual-strong
```

Result:

| kernel | elapsed sec | variance | nonzero | KKT |
|---|---:|---:|---:|---:|
| sparse-residual-strong | 10.330 | 0.01223045 | 251 | 10.903 |

The default auto-search candidates were:

| step | variance | next variance | mean CV log-like |
|---:|---:|---:|---:|
| 0 | 0.01342480 | 0.13424800 | -2632.906 |
| 1 | 0.13424800 | 0.00134248 | -2693.008 |
| 2 | 0.00134248 | 0.01223045 | -2684.009 |

Strong-rule fold counters:

| step | variance | reference fold solves | no-reference fold solves | full-set bypasses | screened nnz | active epochs | full KKT scans |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.01342480 | 0 | 3 | 3 | 0 | 0 | 0 |
| 1 | 0.13424800 | 0 | 3 | 3 | 0 | 0 | 0 |
| 2 | 0.00134248 | 3 | 0 | 0 | 57.5M | 174 | 5 |

Interpretation:

- With the default 10x auto-search step, only the third CV candidate can use a
  compatible historical strong reference. The first candidate has no history;
  the second is a 10x variance increase, which is too far in lambda-space for
  the general strong threshold to be positive.
- On the candidate where strong rules do apply, they screen about `57.5M` nnz
  across the three training folds, roughly `0.67x` one full 3-fold training
  sweep.
- The benefit is mostly cancelled by extra restricted-solve epochs and KKT
  repair. A profiled pair with `CYCLOPS_SPARSE_LOGISTIC_PROFILE=1` gave:

| path | elapsed sec | total CCD epochs | active epochs | GH nnz | GH sec | update nnz | update sec | KKT nnz | KKT sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse-residual | 11.197 | 702 | 0 | 21.56B | 16.855 | 3.95B | 3.437 | 256.9M | 0.199 |
| sparse-residual-strong | 10.604 | 733 | 174 | 19.50B | 15.561 | 3.97B | 3.448 | 352.4M | 0.287 |

Strong rules reduce gradient/Hessian nnz by about `12%`, but total elapsed only
improves about `5-6%` in the profiled run because update work is unchanged, KKT
work increases, and the restricted active solves need more total CCD epochs.
This explains why the default strong-rule path is only a small add-on on 200k.

#### 2x auto-search step diagnostic

Tested whether a tighter auto-search variance step makes historical strong
references useful more often:

```text
CYCLOPS_FOLD_LOCAL_CV=1
CYCLOPS_AUTO_SEARCH_STDSTEP=2
folds=3
threads=3
kernels=sparse-residual,sparse-residual-strong
```

Normal convergence tolerance, no strict final repair:

| kernel | elapsed sec | variance | nonzero | KKT | objective diff vs residual | speedup |
|---|---:|---:|---:|---:|---:|---:|
| sparse-residual | 8.736-8.953 | 0.01649437 | 298 | 80.788 | 0 | 1.00x |
| sparse-residual-strong | 5.980 | 0.01590367 | 317 | 126.245 | -15.770 | 1.46x |

Strong-rule fold counters for the 2x run:

| step | variance | reference fold solves | no-reference fold solves | full-set bypasses | screened nnz | active epochs | full KKT scans |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.01342480 | 0 | 3 | 3 | 0 | 0 | 0 |
| 1 | 0.02684960 | 3 | 0 | 0 | 41.3M | 152 | 4 |
| 2 | 0.00671240 | 3 | 0 | 0 | 70.6M | 189 | 9 |

This confirms the expected mechanics: a tighter auto-search step gives strong
rules compatible references for more candidates, screens much more work
(`112.0M` nnz across CV folds), and can make the run much faster.

However, this is not a retained success criterion. The candidate fold scores
diverge under the current default convergence criterion:

| step | variance | residual mean | strong mean | strong - residual |
|---:|---:|---:|---:|---:|
| 0 | 0.01342480 | -2632.906 | -2632.906 | 0.000 |
| 1 | 0.02684960 | -2634.214 | -2634.612 | -0.399 |
| 2 | 0.00671240 | -2638.043 | -2637.877 | 0.165 |

The search therefore selects a different variance (`0.01590367` vs
`0.01649437`) and the final full-data KKT diagnostic is worse. A measurement-only
active-KKT guard with `CYCLOPS_STRONG_ACTIVE_KKT_TOLERANCE=1e100` showed the
restricted CV folds had modest active KKT values (`max 5.81`), so this is not an
obvious missing inactive repair bug. It is the same broader issue seen in earlier
diagnostics: under Cyclops' default gradient stop, different but valid update
paths can land at different approximate fold solutions and alter auto-search
scores.

The largest step-1 score drift was fold 1 at variance `0.02684960`:

```text
residual training objective: -4796.329, validation log-like: -2769.280, KKT: 6.609
strong training objective:   -4796.215, validation log-like: -2770.473, KKT: 1.707
```

The strong restricted path is more converged by the training objective/KKT
diagnostic there, but it scores worse on the validation fold. This reinforces
that the 2x speed result is not an implementation win by itself; it changes the
effective CV evaluation path under the default stopping rule.

Decision: do not count the 2x strong-rule speedup as goal progress until the
CV/search-path equivalence issue is resolved. It is still useful evidence that
strong rules can help if references are compatible and the restricted solve is
accepted under a convergence criterion that gives comparable fold scores.

Follow-up: tried full sparse-residual polish after each restricted strong solve:

```text
CYCLOPS_FOLD_LOCAL_CV=1
CYCLOPS_AUTO_SEARCH_STDSTEP=2
CYCLOPS_STRONG_FINAL_FULL_POLISH=1
kernel=sparse-residual-strong
```

Result:

| path | elapsed sec | variance | nonzero | KKT |
|---|---:|---:|---:|---:|
| 2x residual baseline | 8.736-8.953 | 0.01649437 | 298 | 80.788 |
| 2x strong + full polish | 6.977 | 0.01596574 | 311 | 117.896 |

The polish did not restore the residual candidate scores:

| step | variance | residual mean | strong full-polish mean | delta |
|---:|---:|---:|---:|---:|
| 0 | 0.01342480 | -2632.906 | -2632.906 | 0.000 |
| 1 | 0.02684960 | -2634.214 | -2634.609 | -0.395 |
| 2 | 0.00671240 | -2638.043 | -2638.018 | 0.025 |

It also added work:

```text
strong full-polish CCD epochs:         731
strong full-polish active epochs:      346
strong full-polish full KKT scans:      21
strong full-polish screened nnz:      111.7M
strong full-polish KKT nnz:           499.0M
```

Decision: reject full-polish as a strong-rule equivalence fix. It remains faster
than the 2x residual run in this single measurement, but it still changes the CV
candidate scores and selected variance while adding repair/polish work. The
strong-rule branch is now best treated as a small retained add-on under the
default 10x auto-search step, not the main route to the remaining 50% goal.

#### Intra-solve active-set convergence diagnostic

Added an env-gated diagnostic for the sparse-residual CCD path:

```text
CYCLOPS_SPARSE_LOGISTIC_ACTIVE_SET=1
CYCLOPS_SPARSE_LOGISTIC_ACTIVE_SET_MAX_FRACTION=<fraction, default 0.50>
CYCLOPS_SPARSE_LOGISTIC_ACTIVE_SET_MAX_PASSES=<passes, default 100>
```

This is different from strong-rule screening. It runs a normal full CCD sweep,
then, if the current nonzero/unpenalized set is small enough, runs active-only
CCD sweeps on those coordinates. If active-only convergence fires, the solver
still runs another full sweep and only accepts convergence after the full pass.
So this diagnostic does not accept an active-only solution.

Implementation note: the diagnostic changes the coordinate path and therefore
can change Cyclops' default approximate CV fold solution. It is not default-on.

50k exact auto-CV, `folds=3`, `threads=3`, fold-local, with hot-loop profiling:

| path | elapsed sec | variance | nonzero | KKT | CCD epochs | active epochs | GH nnz | GH sec | update nnz | update sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline sparse-residual | 2.667 | 0.02600871 | 131 | 1.543 | 672 | 0 | 5.176B | 4.245 | 0.523B | 0.474 |
| active-set diagnostic | 1.326 | 0.02645463 | 134 | 0.130 | 1691 | 1616 | 1.689B | 1.056 | 1.133B | 1.080 |

The active-set path is about `2.0x` faster on 50k under profiling. It cuts
gradient/Hessian work to `33%` of baseline and GH time to `25%`, but it more
than doubles update work. Net wall time still improves because GH is the
dominant cost.

200k exact auto-CV, `folds=3`, `threads=3`, fold-local:

| path | elapsed sec | variance | nonzero | KKT | CCD epochs | active epochs | GH nnz | GH sec | update nnz | update sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| profiled baseline sparse-residual | 11.197 | 0.01228495 | 253 | 10.475 | 702 | 0 | 21.562B | 16.855 | 3.952B | 3.437 |
| profiled active-set diagnostic | 7.025 | 0.01239402 | 258 | 0.813 | 1480 | 1392 | 8.724B | 6.041 | 6.217B | 5.501 |
| unprofiled active-set diagnostic | 6.864 | 0.01239402 | 258 | 0.813 | - | - | - | - | - | - |

Relative to the profiled baseline, active-set convergence reduces GH nnz to
`40%` and GH time to `36%`, while increasing update time to `160%`. Net profiled
wall speedup is `11.197 / 7.025 = 1.59x`; the clean unprofiled timing is
`6.864s`, roughly `10.717 / 6.864 = 1.56x` versus the current untraced
fold-local fitted-response baseline from the AVX2 recheck.

Combining the diagnostic with historical strong rules did not help on 200k:

| path | elapsed sec | variance | nonzero | KKT |
|---|---:|---:|---:|---:|
| active-set diagnostic | 6.864 | 0.01239402 | 258 | 0.813 |
| active-set + strong | 7.205 | 0.01239935 | 258 | 0.846 |

The correctness/search-path caveat remains. The active-set candidate scores
differ from the baseline at the same default 10x auto-search candidates:

| step | variance | baseline mean | active mean | active - baseline |
|---:|---:|---:|---:|---:|
| 0 | 0.01342480 | -2632.906 | -2631.984 | +0.922 |
| 1 | 0.13424800 | -2693.008 | -2692.439 | +0.570 |
| 2 | 0.00134248 | -2684.407 | -2684.593 | -0.186 |

The selected variance changes from `0.01228495` to `0.01239402`, and the final
full-data KKT diagnostic improves substantially (`10.475` to `0.813`). This is
the same pattern as several previous diagnostics: faster and more converged, but
not the same approximate CV path under Cyclops' default gradient stopping rule.

Decision: keep this diagnostic for now. It is the first post-kernel experiment
to exceed the 50% speedup target on 200k, and it does so by attacking the actual
hot path: fewer full-column gradient/Hessian scans. It is not yet a retained
production optimization because CV score/path equivalence is unresolved. The
next work should make this active-set strategy production-safe, either by
defining an acceptable exact convergence criterion for CV folds or by comparing
fixed-candidate replay paths where the selected variance is held constant.

Fixed-lambda full-data check at the baseline selected variance
`0.0122849480391606`:

| tolerance | path | elapsed sec | variance | nonzero | KKT | objective | log-likelihood | CCD epochs | active epochs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-6 | baseline sparse-residual | 3.145 | 0.01228495 | 253 | 10.475 | -10130.653 | -9820.622 | 82 | 0 |
| 1e-6 | active-set diagnostic | 2.553 | 0.01228495 | 252 | 0.881 | -10130.480 | -9820.431 | 231 | 223 |
| 1e-8 | baseline sparse-residual | 11.864 | 0.01228495 | 258 | 0.044 | -10130.452 | -9820.412 | 298 | 0 |
| 1e-8 | active-set diagnostic | 4.759 | 0.01228495 | 252 | 0.068 | -10130.460 | -9820.429 | 475 | 466 |

At fixed variance, the active-set strategy is not just an auto-search artifact.
It is faster on the same fixed problem:

```text
default fixed-lambda speedup: 3.145 / 2.553 = 1.23x
strict fixed-lambda speedup:  11.864 / 4.759 = 2.49x
```

The strict fixed-lambda objective difference is small (`active - baseline =
-0.0081` in the reported objective) and the KKT diagnostics are the same order.
The nonzero count still differs, which is plausible under collinearity but needs
prediction/objective comparison before production use. This strengthens the case
for active-set convergence as the next serious optimization: it materially
reduces full-column scans and becomes more valuable as the requested convergence
tightens.

Strict 200k exact auto-CV check, current build, `folds=3`, `threads=3`,
`tolerance=1e-8`, `maxIterations=5000`:

| path | elapsed sec | variance | nonzero | KKT | objective | log-likelihood | CCD epochs | active epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline sparse-residual | 37.855 | 0.01243368 | 258 | 0.0411 | -10128.59 | -9818.243 | 2415 | 0 |
| active-set diagnostic | 17.656 | 0.01243728 | 252 | 0.0683 | -10128.55 | -9818.208 | 4032 | 3939 |

Speedup:

```text
37.855 / 17.656 = 2.14x
```

Candidate trace comparison under strict convergence:

| step | variance | baseline mean | active mean | active - baseline |
|---:|---:|---:|---:|---:|
| 0 | 0.01342480 | -2632.648 | -2632.543 | +0.105 |
| 1 | 0.13424800 | -2691.997 | -2691.983 | +0.014 |
| 2 | 0.00134248 | -2684.584 | -2684.585 | -0.0008 |

The selected variance changes by only `3.6e-6` (`0.01243368` to
`0.01243728`). This is much cleaner than the default-tolerance auto-CV result:
when the fold solves are forced closer to exact convergence, active-set
convergence preserves the search path closely while giving more than 2x speedup.

Decision update: this is now the strongest candidate for reaching the goal under
the "exact CV, no loosened tolerance" requirement. It still needs repeated
current-build timings and a production decision on the default convergence mode,
but the active-set approach is no longer just a path-sensitivity diagnostic; it
is a real exact-solve accelerator when convergence is strict.

Current verification rerun, same build, no concurrent benchmark jobs, exact
auto-CV with `folds=3`, `threads=3`, `CYCLOPS_FOLD_LOCAL_CV=1`,
`tolerance=1e-8`, and `maxIterations=5000`:

| sample | path | elapsed sec | variance | nonzero | KKT | objective | log-likelihood | CCD epochs | active epochs | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | baseline sparse-residual | 6.696 | 0.02648010 | 138 | 0.0152 | -2732.238 | -2580.016 | 1842 | 0 | 1.00x |
| 50k | active-set diagnostic | 2.120 | 0.02647678 | 134 | 0.0191 | -2732.248 | -2580.021 | 2961 | 2900 | 3.16x |
| 200k | baseline sparse-residual | 39.987 | 0.01243368 | 258 | 0.0411 | -10128.586 | -9818.243 | 2415 | 0 | 1.00x |
| 200k | active-set diagnostic | 17.370 | 0.01243728 | 252 | 0.0683 | -10128.549 | -9818.208 | 4032 | 3939 | 2.30x |

This rerun confirms the strict active-set result on 200k and adds a matching
50k strict CV check. The active-set path remains above the 50% target on both
sample sizes. The selected variance is very close to baseline; objective and
log-likelihood differences are small, but the active-set path still returns a
slightly different sparse support. Treat this as a strong optimization candidate
that needs production hardening and prediction-difference checks, not as a final
default-on change yet.

Initially added a benchmark-only alias:

```text
--kernels sparse-residual,sparse-residual-active
```

That alias enabled `CYCLOPS_SPARSE_LOGISTIC_ACTIVE_SET=1` only around that fit
and passes `sparse-residual` to Cyclops. This allows baseline and active-set
fits to run in the same R process, so the benchmark script can report prediction
RMSE and max absolute prediction differences.

Strict in-process validation, same settings:

| sample | path | elapsed sec | variance | nonzero | KKT | objective diff | log-likelihood diff | prediction RMSE | prediction max abs diff | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | sparse-residual | 7.928 | 0.02648010 | 138 | 0.0152 | 0 | 0 | 0 | 0 | 1.00x |
| 50k | sparse-residual-active | 2.263 | 0.02647678 | 134 | 0.0191 | -0.00953 | -0.00535 | 1.30e-05 | 6.88e-04 | 3.50x |
| 200k | sparse-residual | 42.909 | 0.01243368 | 258 | 0.0411 | 0 | 0 | 0 | 0 | 1.00x |
| 200k | sparse-residual-active | 18.794 | 0.01243728 | 252 | 0.0683 | 0.0367 | 0.0350 | 1.45e-05 | 0.00600 | 2.28x |

This gives direct prediction-difference evidence for the strict active-set path.
The prediction RMSE is around `1e-5` on both matrix sizes, while wall-clock
speedup remains above `2x`. The remaining production questions are how to expose
the option cleanly outside an env var and whether Cyclops should keep the current
default gradient stopping behavior or require stricter convergence for this
path.

Follow-up hardening: replaced the benchmark-only alias with a real Cyclops
control value:

```r
Cyclops::createControl(logisticKernel = "sparse-residual-active")
```

Internally this maps to `LogisticKernelType::SPARSE_RESIDUAL_ACTIVE`. It uses
the same fitted-response sparse gradient/update kernels as `sparse-residual`,
but enables the intra-solve active-set convergence loop without requiring
`CYCLOPS_SPARSE_LOGISTIC_ACTIVE_SET`. The environment variable is still accepted
as an experimental override for plain `sparse-residual`.

Strict public-kernel validation, same settings:

| sample | path | elapsed sec | variance | nonzero | KKT | objective diff | log-likelihood diff | prediction RMSE | prediction max abs diff | CCD epochs | active epochs | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | sparse-residual | 7.858 | 0.02648010 | 138 | 0.0152 | 0 | 0 | 0 | 0 | 1842 | 0 | 1.00x |
| 50k | sparse-residual-active | 2.296 | 0.02647678 | 134 | 0.0191 | -0.00953 | -0.00535 | 1.30e-05 | 6.88e-04 | 2961 | 2900 | 3.42x |
| 200k | sparse-residual | 44.117 | 0.01243368 | 258 | 0.0411 | 0 | 0 | 0 | 0 | 2415 | 0 | 1.00x |
| 200k | sparse-residual-active | 19.112 | 0.01243728 | 252 | 0.0683 | 0.0367 | 0.0350 | 1.45e-05 | 0.00600 | 4032 | 3939 | 2.31x |

This confirms the active-set speedup through the user-facing control path. The
next hardening step is implementation cleanup: reuse the active-index buffer
instead of allocating it inside every active pass, add explicit active-set size
counters, and then rerun the same strict public-kernel validation.

Implemented the first cleanup by reusing one `activeIndices` buffer in the
non-QN loop instead of allocating a new vector for every active pass. This is not
intended to change convergence behavior; it just removes avoidable allocation
work.

Post-cleanup strict validation:

| sample | path | elapsed sec | variance | nonzero | KKT | objective diff | prediction RMSE | prediction max abs diff | CCD epochs | active epochs | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | sparse-residual | 7.457 | 0.02648010 | 138 | 0.0152 | 0 | 0 | 0 | 1842 | 0 | 1.00x |
| 50k | sparse-residual-active | 2.297 | 0.02647678 | 134 | 0.0191 | -0.00953 | 1.30e-05 | 6.88e-04 | 2961 | 2900 | 3.25x |
| 200k | sparse-residual | 44.223 | 0.01243368 | 258 | 0.0411 | 0 | 0 | 0 | 2415 | 0 | 1.00x |
| 200k | sparse-residual-active | 18.985 | 0.01243728 | 252 | 0.0683 | 0.0367 | 1.45e-05 | 0.00600 | 4032 | 3939 | 2.33x |

The cleanup preserves the same selected variances, support sizes, KKT values,
prediction differences, and active/full epoch counts. Any elapsed change is
within normal run-to-run noise.

Added active-set size instrumentation for the active convergence loop. The
existing `sparseLogistic.activeSetSizeMean` / `setSizeRecords` fields now record
the active coordinate count for each actual active-only sweep in
`sparse-residual-active`.

Strict validation after instrumentation:

| sample | path | elapsed sec | variance | nonzero | KKT | objective diff | prediction RMSE | prediction max abs diff | CCD epochs | active epochs | active set mean | active set records | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | sparse-residual | 7.217 | 0.02648010 | 138 | 0.0152 | 0 | 0 | 0 | 1842 | 0 | 0.0 | 0 | 1.00x |
| 50k | sparse-residual-active | 2.192 | 0.02647678 | 134 | 0.0191 | -0.00953 | 1.30e-05 | 6.88e-04 | 2961 | 2900 | 160.0 | 2900 | 3.29x |
| 200k | sparse-residual | 41.741 | 0.01243368 | 258 | 0.0411 | 0 | 0 | 0 | 2415 | 0 | 0.0 | 0 | 1.00x |
| 200k | sparse-residual-active | 18.653 | 0.01243728 | 252 | 0.0683 | 0.0367 | 1.45e-05 | 0.00600 | 4032 | 3939 | 374.2 | 3939 | 2.24x |

This confirms the active-set speedup is coming from many active-only sweeps over
small coordinate sets: roughly 160 active coordinates on 50k and 374 on 200k,
versus 14k-17k total covariates.

Repeated stability check, same strict public-kernel settings,
`repetitions=3`, no concurrent benchmark jobs:

| sample | baseline median sec | active median sec | median speedup | min speedup | max speedup | active variance | active nonzero | active KKT | pred RMSE | pred max abs diff | active epochs | active set mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | 6.933 | 2.060 | 3.36x | 3.31x | 3.39x | 0.02647678 | 134 | 0.0191 | 1.30e-05 | 6.88e-04 | 2900 | 160.0 |
| 200k | 40.379 | 18.453 | 2.19x | 2.19x | 2.22x | 0.01243728 | 252 | 0.0683 | 1.45e-05 | 0.00600 | 3939 | 374.2 |

This satisfies the current stability target for the strict exact-CV benchmark:
the worst repeated 200k speedup is still `2.19x`, comfortably above the required
`1.5x` threshold relative to the fitted-response sparse-residual baseline.
Remaining caveat: the active path is validated under strict convergence. Under
Cyclops' default tolerance, it remains faster but changes the approximate CV
path more visibly, so the production recommendation should either document this
as a strict/exact-CV accelerator or make the convergence contract explicit.

## Lazy selective-polish default-tolerance active-set status

Follow-up work tested a lazy exactness strategy for the default-tolerance CV
path:

1. Use `sparseConvergenceType = "fitted-change"` and `tolerance = 1e-6` for
   most exploratory fold solves.
2. Polish decision-sensitive auto-search candidates at `cvPolishTolerance =
   1e-8` using the standard sparse convergence criterion.
3. Polish the selected final full-data fit again, with
   `CYCLOPS_FINAL_POLISH_TOLERANCE=1e-9` in these measurements.

Implementation hardening:

- `UniModalSearch::tried()` now recomputes the cached best point when a polished
  score replaces an existing loose score.
- All CV/final polish passes now switch `sparseConvergenceType` to `"auto"` so
  polish means a stricter standard convergence criterion, not just a smaller
  fitted-change threshold.
- Selective polish now polishes the initial candidate by default and keeps
  geometry repair enabled for unpolished bracket neighbors that can affect the
  search path. This is controlled by
  `cvPolishInitialCandidates` in `createControl()`, and by
  `--cv-polish-initial-candidates` in the benchmark wrapper. The environment
  variable `CYCLOPS_CV_SELECTIVE_POLISH_MIN_INITIAL` remains available as an
  override; the default is `1`. The geometry repair can be disabled with
  `CYCLOPS_CV_SELECTIVE_POLISH_REPAIR_GEOMETRY=0`.
- After a polished stop-verification candidate, selective polish also evaluates
  one final proposed stopping variance with polished CV if auto-search proposes
  an unevaluated finite stopping point. This is controlled by
  `cvPolishStopProposal` in `createControl()`, and by
  `--cv-polish-stop-proposal` in the benchmark wrapper. The environment
  variable `CYCLOPS_CV_SELECTIVE_POLISH_STOP_PROPOSAL=0` remains available as
  an override.
- Local reflected refinement remains adaptive. By default it only runs if the
  verified stop point is materially worse than the best polished point. Setting
  `CYCLOPS_CV_SELECTIVE_POLISH_LOCAL_REFINE_RELATIVE_WINDOW=0` forces any
  improvement to trigger the reflected candidate.

Latest non-concurrent 3-fold / 3-thread lazy selective-polish comparisons with
the initial-candidate plus geometry-repair default and stop-proposal polish,
rerun after the explicit R/Rcpp control surface was added:

| sample | path | elapsed sec | variance | nonzero | KKT | objective diff | prediction RMSE | prediction max abs diff | CCD epochs | active epochs | full reconciliation sweeps | active set mean | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | sparse-residual | 10.538 | 0.02647111 | 138 | 0.0152 | 0 | 0 | 0 | 3603 | 0 | 0 | 0.0 | 1.00x |
| 50k | sparse-residual-active | 4.115 | 0.02648248 | 134 | 0.000674 | 0.0327 | 5.09e-06 | 1.54e-04 | 6337 | 6178 | 59 | 130.7 | 2.56x |
| 200k | sparse-residual | 68.123 | 0.01342480 | 265 | 0.0391 | 0 | 0 | 0 | 4941 | 0 | 0 | 0.0 | 1.00x |
| 200k | sparse-residual-active | 31.441 | 0.01342480 | 262 | 0.00666 | 4.94e-06 | 1.88e-06 | 4.73e-05 | 8135 | 7962 | 56 | 315.3 | 2.17x |

Polishing only the initial candidate plus repairing bracket-sensitive loose
candidates reduced the earlier active/non-active search-path drift while
preserving more loose exploratory work than full initial-bracket polishing. On
50k, the active selected variance moved from the older drifted `0.02698589` to
`0.02648248`, close to the residual path's `0.02647111`. On 200k, the active
and residual paths selected the same variance, `0.01342480`. The speedup remains
about `2.6x` on 50k and `2.2x` on 200k.

Fold traces confirm the tradeoff: loose exploratory candidates can have fold KKT
residuals around `1-4`, but polished candidates are much tighter. For example,
on the 50k active fold trace, the polished accepted candidate had fold KKT
values `0.00048`, `0.00273`, and `0.00123`; loose bracket candidates had much
larger KKT values. This is expected for lazy exactness and means the mode should
be presented as fast exploratory CV with reliable polished decision points and a
strict final fit, not as exact CV for every evaluated loose candidate.

The candidate trace records `polish_reason`. On the latest 50k active sanity
run, the auto-search sequence was:

| step | variance | polished | polish reason |
|---:|---:|---:|---|
| 0 | 0.01549469 | yes | `initial` |
| 1 | 0.15494691 | no | `none` |
| 2 | 0.00154947 | yes | `initial-bracket` |
| 3 | 0.15494691 | yes | `repair-existing` |
| 4 | 0.02648248 | yes | `stop-verify` |
| 5 | 0.02624968 | yes | `stop-proposed` |

This is the intended safer lazy path: exploratory work is still mostly cheap,
but bracket-sensitive and final proposed stopping points are scored with
polished fold fits before they influence the accepted variance.

The active path now records explicit full-sweep reconciliation before returning
from active-only convergence. On the same 50k, 3-fold / 3-thread active sanity
run after a clean native rebuild, the current default benchmark reported:

| metric | value |
|---|---:|
| elapsed seconds | 4.115 |
| selected variance | 0.02648248 |
| nonzero coefficients | 134 |
| max KKT violation | 0.000674 |
| total CCD epochs | 6337 |
| active-only epochs | 6178 |
| full reconciliation sweeps | 59 |
| mean active set size | 130.7 |

This verifies the production shape intended for active-set scoring: active-only
sweeps do most of the work, but a full all-coordinate sweep/check is still run
before the solver accepts active-only convergence for a fold/candidate.

The fold trace now exposes the same counter as
`active_set_full_reconciliation_sweeps`, so this can be checked per scored fold.
In a traced 50k active run with the same settings, the high-side bracket
candidate was first scored loosely and then repaired once it became
bracket-sensitive. Across the 18 scored fold/candidate rows, every row with
active-only epochs also had at least one full reconciliation sweep:

| step | variance | active-only epochs | full reconciliation sweeps |
|---:|---:|---:|---:|
| 0 | 0.01549469 | 1125 | 11 |
| 1 | 0.15494691 | 350 | 9 |
| 2 | 0.00154947 | 583 | 9 |
| 3 | 0.15494691 | 1321 | 10 |
| 4 | 0.02648248 | 1097 | 10 |
| 5 | 0.02624968 | 1244 | 5 |

The explicit check for `active_set_only_epochs > 0` and
`active_set_full_reconciliation_sweeps == 0` returned zero rows.

As a control, a lighter selective-polish policy with geometry repair disabled
was also tested on 50k. This allowed one initial bracket candidate to be scored
from the loose exploratory fit and never repaired:

| step | variance | polished | polish reason |
|---:|---:|---:|---|
| 0 | 0.01549469 | yes | `initial` |
| 1 | 0.15494691 | no | `none` |
| 2 | 0.00154947 | yes | `initial-bracket` |
| 3 | 0.02698589 | yes | `stop-verify` |
| 4 | 0.02675285 | yes | `stop-proposed` |

That run was faster (`2.963s`, 4867 CCD epochs, 4753 active-only epochs, 42
full reconciliation sweeps), and every active fold solve still had at least one
full reconciliation sweep before scoring. However, it selected the older drifted
variance `0.02698589` rather than the safer full-bracket-polished variance
`0.02648145`. With geometry repair enabled by default, the same
`--cv-polish-initial-candidates 1` policy repairs that unpolished bracket
neighbor (`polish_reason = repair-existing`) and selects `0.02648248` on 50k.
On 200k, the same default geometry-repair policy leaves the high-side bracket
loose initially, repairs it before it can distort the bracket, and keeps the
selected variance at `0.01342480`.

Disabling the low-side initial-bracket polish with
`CYCLOPS_CV_SELECTIVE_POLISH_INITIAL_BRACKET=0` was also tested on 50k. That
allowed both bracket sides to be loose initially and was faster (`3.078s`, 4813
CCD epochs, 4691 active-only epochs, 47 full reconciliation sweeps), but it
again selected the drifted `0.02698589` variance. So the current default keeps
the low-side initial bracket polished while relying on geometry repair for the
high-side bracket.

The selected final full-data fit is also now traceable with
`CYCLOPS_FINAL_POLISH_TRACE`. In a 50k, 3-fold / 3-thread active sanity run with
`CYCLOPS_FINAL_POLISH_TOLERANCE=1e-9` and
`CYCLOPS_FINAL_POLISH_MAX_ITERATIONS=1500`, the final selected-variance fit
first ran the exploratory settings and then polished:

| stage | tolerance | max iterations | sparse convergence | iterations | convergence |
|---|---:|---:|---|---:|---:|
| base final fit | 1e-6 | 1000 | `fitted-change` | 67 | 9.73e-07 |
| final polish | 1e-9 | 1500 | `auto` | 404 | 9.85e-10 |

This directly verifies that the selected model used for final coefficients and
predictions receives the stricter polish, not only the CV fold candidates.

A 50k check with
`sparseConvergenceType = "coordinate-impact"` was slower (`7.31s` residual,
`2.68s` active) and had larger objective drift (`0.960`), so `fitted-change`
remains the better loose exploratory criterion in these tests.
