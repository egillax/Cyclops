# Cyclops sparse logistic performance review

This document reorganizes the performance work from
`docs/cyclops-sparse-logistic-performance.md` for external review. It separates:

1. the changes that worked;
2. the experiments that were tried and rejected;
3. the remaining plausible work;
4. questions where another performance/modeling review would be useful.

The main benchmark target throughout was exact 3-fold CV on OHDSI/PLP sparse
logistic data, using 3 threads and non-concurrent benchmark runs. Single-thread
benchmarks are no longer considered decision-relevant.

## Executive Summary

The largest proven improvement is the fitted-response sparse logistic kernel
plus active-set/lazy-polish CV path.

The original fitted-response kernel work replaced the heavier legacy logistic
state with a simpler probability state:

```cpp
fitted[i] = Pr(y_i = 1 | beta)
```

For an indicator sparse column, the hot gradient/Hessian loop became:

```cpp
for row in rows_j:
    p = fitted[row]
    g += p
    h += p * (1 - p)
g -= xjY[j]
```

The later active-set path did not replace CCD with glmnet-style IRLS. It kept
Cyclops' coordinate-descent structure, but allowed most inner sweeps to run on a
small active coordinate set, with explicit full-coordinate reconciliation before
accepting active-only convergence.

The current best default-tolerance strategy is lazy selective polish:

```text
exploratory CV folds:
    sparseConvergenceType = fitted-change
    tolerance = 1e-6

decision-sensitive candidates:
    sparseConvergenceType = auto
    cvPolishTolerance = 1e-8

selected final full-data fit:
    strict final polish
```

Latest non-concurrent 3-fold / 3-thread comparison:

| sample | path | elapsed sec | variance | nonzero | KKT | objective diff | prediction RMSE | prediction max abs diff | CCD epochs | active epochs | full reconciliation sweeps | active set mean | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | sparse-residual | 10.538 | 0.02647111 | 138 | 0.0152 | 0 | 0 | 0 | 3603 | 0 | 0 | 0.0 | 1.00x |
| 50k | sparse-residual-active | 4.115 | 0.02648248 | 134 | 0.000674 | 0.0327 | 5.09e-06 | 1.54e-04 | 6337 | 6178 | 59 | 130.7 | 2.56x |
| 200k | sparse-residual | 68.123 | 0.01342480 | 265 | 0.0391 | 0 | 0 | 0 | 4941 | 0 | 0 | 0.0 | 1.00x |
| 200k | sparse-residual-active | 31.441 | 0.01342480 | 262 | 0.00666 | 4.94e-06 | 1.88e-06 | 4.73e-05 | 8135 | 7962 | 56 | 315.3 | 2.17x |

On 200k, active and baseline selected the same variance. On 50k, the selected
variance was close but not identical. The active final fit had much tighter KKT
than the baseline in these runs.

The most important caveat is semantic: the lazy path is not exact CV for every
exploratory candidate. It is fast exploratory CV with strict polish on candidates
that can affect the final decision, plus a strict final selected fit.

## Current Hot Path

The current sparse residual/fitted-response kernel is still dominated by sparse
gathers:

```cpp
p = fitted[row]
```

Earlier profiling consistently showed that the remaining cost is not arithmetic
alone. It is sparse indexed access into the row-state vector and sparse state
updates. Cache behavior remained meaningfully worse than glmnet-style tight
sparse dot-product loops.

Representative retained perf interpretation after fold-local/lazy setup work:

- setup overhead was largely removed;
- core cycles dropped versus non-fold-local/fold-local eager variants;
- L1D behavior improved in some runs;
- LLC miss rate remained high;
- the remaining hot work is sparse gather/update, not CV scoring or fold setup.

This is the key design pressure for any next optimization: reduce the number of
times `fitted[row]` is gathered, or make those gathers materially more local.

## Implemented And Positive

### 1. Fitted-response sparse logistic kernel

Status: retained.

Main idea:

```text
legacy:
    maintain richer logistic state: xBeta, exp(xBeta), denominator/cache-like state
    compute gradient/Hessian through generic logistic machinery
    update multiple row-state vectors

fitted-response sparse kernel:
    maintain fitted probability p[i]
    compute sparse logistic gradient/Hessian directly from p[i]
    specialize indicator columns
    update p[i] after coordinate changes
```

Why it helps:

- removes unnecessary per-row logistic state from the hot loop;
- avoids repeated exp/denominator work in coordinate gradient/Hessian;
- lets indicator columns run through a very simple sparse reduction;
- gives Cyclops a hot loop closer to sparse column dot/reduce work.

This is the core change that made later active-set work profitable. Earlier
restricted/active-set attempts on the legacy kernel did not produce the same
gain because the per-coordinate state work was still too expensive.

### 2. Fold-local compact path with lazy sparse-index setup

Status: partially retained as a useful setup/layout improvement.

Fold-local compact CSC was tested to replace global row ids and weighted
training-row views with compact fold-local row ids. The naive/R-level and early
native versions had high setup cost and some solver-path/convergence differences.

The important positive sub-result was skipping legacy sparse-index
materialization for sparse logistic fold-local data:

| sample | path | elapsed sec | variance | nonzero | KKT | build sec | data build sec | CCD setup sec | init sec | GH sec | update sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | fold-local baseline | 4.818 | 0.02600871 | 131 | 1.543 | 1.353 | 0.194 | 1.159 | - | 4.201 | 0.490 |
| 50k | fold-local + skip sparse indices | 2.688 | 0.02600871 | 131 | 1.543 | 0.194 | 0.191 | 0.003 | 0.003 | 4.214 | 0.485 |
| 200k | fold-local baseline | 21.388 | 0.01228495 | 253 | 10.475 | 6.153 | 0.814 | 5.339 | 5.338 | 17.516 | 3.459 |
| 200k | fold-local + skip sparse indices | 11.293 | 0.01228495 | 253 | 10.475 | 0.786 | 0.774 | 0.011 | 0.010 | 17.061 | 3.348 |

Interpretation:

- The fold-local setup tax was mostly caused by legacy sparse-index
  materialization that the fitted-response sparse logistic path did not need.
- Removing it made fold-local viable from a setup perspective.
- It did not remove the remaining hot sparse gather/update cost.

### 3. Historical strong-rule screening

Status: small positive add-on, not the main path.

The original monotone immediate-previous sequential strong rule did not fit
Cyclops auto-search well because auto-search moves nonmonotonically in variance.

The improved version used:

```text
historical per-fold reference bank
general threshold:
    threshold_j = tau_current_j - abs(tau_current_j - tau_ref_j)
exact full KKT repair
full-set bypass when nothing is screened
```

This helped in some auto-search steps, but the benefit was limited because:

- many candidates still had no compatible reference;
- screened nnz was not consistently enough to dominate;
- restricted solves could require more epochs and repair scans.

Same-process cleaned check:

| sample | kernel | elapsed sec | variance | nonzero | KKT | objective diff vs residual | speedup vs residual |
|---|---|---:|---:|---:|---:|---:|---:|
| 50k | sparse-residual | 2.297 | 0.02600871 | 131 | 1.543 | 0 | 1.00x |
| 50k | sparse-residual-strong | 1.999 | 0.02600872 | 131 | 1.543 | 2.24e-05 | 1.15x |
| 200k | sparse-residual | 10.379 | 0.01228495 | 253 | 10.475 | 0 | 1.00x |
| 200k | sparse-residual-strong | 9.761 | 0.01223045 | 251 | 10.903 | -0.696 | 1.06x |

Decision:

- keep as diagnostic/small add-on only;
- do not expect it to deliver the remaining large speedup by itself.

### 4. Intra-solve active-set convergence

Status: most promising production optimization after fitted-response kernel.

This is not strong-rule screening. It runs normal CCD, but once the solver has a
small active/nonzero set, many sweeps are restricted to active coordinates. Before
accepting active-only convergence, the solver runs a full all-coordinate
reconciliation sweep/check.

Strict convergence validation showed stable speedups:

| sample | baseline median sec | active median sec | median speedup | min speedup | max speedup | active variance | active nonzero | active KKT | pred RMSE | pred max abs diff | active epochs | active set mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | 6.933 | 2.060 | 3.36x | 3.31x | 3.39x | 0.02647678 | 134 | 0.0191 | 1.30e-05 | 6.88e-04 | 2900 | 160.0 |
| 200k | 40.379 | 18.453 | 2.19x | 2.19x | 2.22x | 0.01243728 | 252 | 0.0683 | 1.45e-05 | 0.00600 | 3939 | 374.2 |

The current default-tolerance lazy-polish version preserves most of this benefit
while controlling auto-search drift with polished decision points.

### 5. Lazy selective polish for auto-CV

Status: current best default-tolerance strategy.

Purpose:

- avoid making every exploratory CV fold solve strict/expensive;
- make decision-sensitive candidates reliable;
- preserve normal final selected-model exactness.

Current default policy:

- polish one initial candidate by default;
- keep low-side initial bracket polished;
- geometry repair default-on for loose bracket neighbors that can affect search;
- polish stop-verification candidate;
- polish one final proposed stopping variance;
- strict final selected full-data fit polish.

Important implementation details:

- `UniModalSearch::tried()` recomputes cached best after polished score
  replacement.
- CV/final polish switches sparse convergence type to `auto`.
- `active_set_full_reconciliation_sweeps` is recorded in fold traces.
- Every active fold solve in the traced 50k run had at least one full
  reconciliation sweep before acceptance.

Latest default-tolerance result is the table in the executive summary.

## Tried And Rejected Or Deprioritized

### Row-wise inactive KKT scanner

Status: not implemented as production path.

Reason:

- KKT scan time was not the bottleneck in the observed strong/active runs.
- Full KKT scan time was small relative to gradient/Hessian and update work.
- A row-wise scanner would not reduce the dominant active CCD scans.

Decision:

- only revisit if a future screening method creates very large inactive sets and
  KKT scans become a measurable bottleneck.

### Exact duplicate binary-column compression

Status: diagnostic helpers kept; production compression rejected for now.

Diagnostics found many duplicate groups, so the idea had scan-reduction
potential. However, fitting compressed columns and expanding coefficients back
did not preserve objective/KKT closely enough under auto-CV.

Example fixed-variance diagnostic:

```text
original fixed: 5.395s, objective -10130.65, loglik -9820.622, KKT 10.48
dedup fixed:   3.075s, objective -10139.93, loglik -9822.562, KKT 114.35
```

Later stricter checks improved some fixed-variance behavior, but auto-CV still
selected different points and expanded-original KKT residuals were problematic.

Decision:

- not a near-term production optimization;
- revisit only with a stronger coefficient expansion/equivalence design.

### Basic GAP-safe screening

Status: diagnostic-only; do not wire into solver.

Basic GAP-safe was attractive because it is compatible with nonmonotone
auto-search in principle. The problem was operational: it screened the wrong
columns for this PLP workload.

Observed diagnostic:

```text
sparse-residual:
    safe-screened columns: 13.8-15.3% of eligible zero columns
    safe-screened nnz:     66,491 / 260,192,745 = 0.026%

sparse-residual-strong:
    safe-screened nnz:     35,791 / 259,357,733 = 0.014%

diagnostic pass cost:
    about 29-30 seconds
```

The relevant metric is nnz share, not column share, because the hot path cost is
mostly sparse `fitted[row]` gathers. A 0.026% nnz reduction would require
thousands of future full scans to repay even one expensive screening pass.

Critical-radius and over-solve diagnostics supported the same conclusion:

- safe radius remained too large at normal tolerance;
- screened high-nnz columns did not appear cheaply enough;
- reducing the radius enough would require too much extra convergence work.

Decision:

- close basic/dynamic GAP-safe for now;
- only revisit through a materially stronger celer-style dual extrapolation or
  working-set design.

### Fold-local compact CSC beyond setup fixes

Status: partially positive, not a complete production answer.

What worked:

- compact fold data can reduce some core work;
- skipping legacy sparse-index setup removed a large fold-local setup tax.

What did not fully work:

- naive R materialization was too expensive;
- compact validation clones doubled build work and slowed total runtime;
- row labels were not the key cost;
- copy-fitted / warm `X beta` did not materially help;
- row reordering by row density/outcome did not provide reliable gains;
- fold-local selected slightly different CV paths under default convergence.

Decision:

- keep the setup lessons;
- do not treat fold-local compact CSC as solved production work unless the
  convergence semantics and path differences are explicitly accepted.

### IRLS / WLS / Gram attempts

Status: diagnostics rejected as general replacement.

All-feature sparse WLS residual-CD was much better than naive diagonal/Jacobi
style updates, but still much slower than fitted-response CCD when driven to
exact KKT.

Representative diagnostics:

| sample | method | setup/build sec | solve sec | final KKT | final nonzero |
|---|---|---:|---:|---:|---:|
| 50k | fitted-response CCD | - | 0.687 | 9.44e-7 | 104 |
| 50k | sparse WLS residual-CD, 26 x 20 epochs | 0.090 | 4.351 | 9.44e-7 | 104 |
| 200k | fitted-response CCD | - | 1.249 | 0.0130 | 180 |
| 200k | sparse WLS residual-CD, 30 x 20 epochs | 0.370 | 25.037 | 0.0130 | 180 |

Active-set Gram diagnostics showed dense solve time was tiny, but Gram
construction dominated unless the active set was very small and stable.

Decision:

- do not implement all-feature IRLS/WLS in production;
- a narrow late-path active-set WLS/Gram accelerator remains an open research
  idea, but only if the active set is small, stable, and expensive enough to
  justify Gram construction.

### Row-pass / CSR full-gradient and diagonal Newton diagnostics

Status: rejected as drop-in replacement.

CSR row-pass can load row state once and scatter to feature gradients, which is
conceptually attractive for reducing repeated `fitted[row]` gathers. But the
tested full-batch diagonal/IRLS/proximal variants were too weak:

- too many coefficients activated;
- objective/KKT were far from CCD after reasonable work;
- not competitive without a stronger inner solver or working set.

Decision:

- row-pass remains useful as a building block for KKT/violation scoring, IRLS
  setup, or a future block/working-set method;
- it is not a direct replacement for current CCD.

### Kernel micro-optimizations

Status: mostly small or negative.

Tried:

- AVX2 gather for indicator gradient/Hessian;
- scalar unroll variants;
- prefetch variants;
- coordinate ordering by nnz;
- no-op sparse update guard;
- gradient-first zero-coordinate skip;
- GH-fitted cache for immediate update;
- all-unit fold-local view fast path;
- sparse-state ready guard fast path;
- sparse KKT swindle variant.

Outcomes:

- AVX2 gather gave at most small gains in some runs and was not a main path.
- x8 unroll was slower than the existing x4/scalar shape.
- unconditional prefetch retry was worse than the cleaned baseline.
- coordinate ordering changed convergence/search path and KKT behavior.
- gradient-first zero-coordinate skip could scan columns twice and did not help.
- guard/branch cleanups were too small or negative.

Decision:

- do not chase more micro-patches until the algorithm/data layout changes enough
  to make a new hot loop.

### Compiler flags

Status: not a useful project-level optimization.

Native compiler flag experiments were rejected as a main path. The important
performance differences came from algorithm/kernel/data layout choices, not from
custom flags alone.

## Remaining Plausible Work

These are not untried checkboxes from the original plan. They are the remaining
ideas that still look technically plausible after the experiments.

### A. Production hardening and review splitting

This is the highest-priority engineering step if the goal is a PR.

The current branch is broad and experimental. Split the work into reviewable
pieces:

1. fitted-response sparse logistic kernel core;
2. active-set convergence loop and reconciliation counters;
3. lazy selective-polish CV controls;
4. benchmark tooling and diagnostics;
5. docs.

Strip or gate experimental code that should not ship:

- rejected GAP-safe production hooks;
- rejected fold-local variants;
- rejected micro-kernel env paths;
- one-off benchmark-only diagnostics.

Suggested PR positioning:

```text
This keeps Cyclops' sparse logistic CCD algorithm, but adds a fitted-response
sparse residual kernel and an opt-in active-set/lazy-polish CV mode. The active
mode uses cheap exploratory CV solves and strict polish on decision-sensitive
candidates and the final selected fit.
```

### B. Validate lazy-polish semantics on more data

The current default-tolerance active path is promising but should be tested on
more PLP datasets before making it default.

Need:

- multiple outcomes/cohorts;
- larger samples/full datasets;
- same folds and same benchmark wrapper;
- report selected variance, prediction RMSE, max prediction diff, KKT, objective
  diff, support size, and runtime.

Key question:

```text
Does lazy selective polish keep auto-CV decisions stable enough across real PLP
workloads while preserving the 2x-ish speedup?
```

### C. Stronger active-set exactness contract

The active path now runs full-coordinate reconciliation sweeps before accepting
active-only convergence, and fold traces expose the counter.

Remaining review question:

```text
Is the reconciliation criterion sufficient as a production exactness contract,
or should active-set convergence require an explicit KKT/subgradient bound rather
than relying on the standard Cyclops convergence calculation plus full sweep?
```

This matters because different convergence criteria changed speed and selected
variance in earlier tests.

### D. Celer-style dual extrapolation / working-set solver

Basic GAP-safe was negative, but celer-style methods are different:

```text
working set + dual extrapolation + tighter certificates + exact repair
```

This remains plausible only as a larger algorithmic project. It would need:

- feasible dual construction with Cyclops intercept/weights/folds;
- dual extrapolation across late CCD iterates;
- nnz-weighted screening diagnostics, not just column counts;
- exact KKT repair;
- clear break-even against the current active sparse-residual path.

Success bar:

```text
screen >5-10% of eligible nnz, including hot columns, early enough to reduce
total CV time.
```

### E. Deeper data-layout work targeting fitted-row gathers

The remaining hot path is still:

```cpp
p = fitted[row]
```

Simple fold-local compact CSC helped setup and sometimes core cycles, but did
not eliminate the gather bottleneck. More invasive options:

- block-coded row indices;
- row-index compression with cache-sized row blocks;
- split kernels by column frequency bucket;
- hot-column dense/bitset representation;
- CSR/CSC hybrid for violation scoring or block updates;
- row clustering based on feature co-occurrence, not just row nnz/outcome.

These should be evaluated with:

- cycles per nnz;
- L1D/LLC misses per nnz;
- dTLB misses per nnz;
- solve time excluding build;
- total time including build;
- selected variance/objective/KKT/prediction diffs.

The bar should be a real total-time improvement over the current active path,
not just lower kernel time in isolation.

### F. Narrow active-set WLS/Gram accelerator

General WLS/IRLS was not competitive. A narrow accelerator might still help if
used only when:

- active set is small and stable;
- active support is in the low hundreds;
- current CCD would otherwise spend many active-only epochs;
- exact fitted-response CCD/KKT repair follows the accelerator.

Possible design:

```text
if active_set_size <= threshold and active_set_stable:
    build WLS/Gram on active set
    run a few dense/covariance CD steps
    map beta back
    continue exact fitted-response active CCD
    full KKT repair before acceptance
else:
    use current active sparse-residual path
```

Open risk:

- Gram build can dominate.
- Auto-CV active sets may be less stable than final-fit active sets.

### G. Full glmnet-like path redesign

The current solver is still Cyclops CCD, not glmnet's full GLM path design.

A true glmnet-like redesign would mean:

- pathwise lambda ordering;
- strong rules/ever-active sets as first-class path machinery;
- IRLS/quadratic approximation for GLMs;
- sparse residual dot/update inner loops;
- exact repair and KKT checking;
- auto-search adapted to a pathwise continuation model.

This is a large project and probably not the right next PR, but it is the
strategic route if the goal is to compete directly with glmnet across many
logistic CV workloads.

## How Active Sets Change Earlier Ideas

The active-set result changes the evaluation of earlier approaches, but not
equally.

Short version:

```text
strong rules:       worth retesting on top of active sets
GAP-safe:           only worth retesting as a diagnostic/late-polish aid
fold-local/layout:  still plausible, but must beat active baseline
IRLS/WLS/Gram:      more plausible only as a narrow active-set accelerator
```

The reason active sets worked is that they changed the structure from:

```text
screen first, then hope the reduced problem is enough
```

to:

```text
solve mostly on the small active/nonzero set,
then explicitly reconcile against the full coordinate set before accepting
```

That pattern is also the only way the earlier screening and WLS ideas are likely
to become useful: cheap restricted progress, explicit full-coordinate
reconciliation, and strict polish where CV decisions matter.

### Strong rules after active sets

Strong rules should be retested as an initializer or outer candidate filter for
the active path, not as a replacement for it.

Earlier problems:

1. Cyclops auto-search is nonmonotone, so immediate-previous sequential rules
   often degenerated to full sets.
2. Restricted strong-set solves could spend extra work before KKT repair found
   missing coordinates.

The active-set machinery helps with the second problem. A better combined
structure is:

```text
for each fold/lambda:
    build candidate set C from:
        current warm-start nonzeros
        previous/reference nonzeros
        unpenalized/intercept
        historical-reference strong-rule candidates

    if C is full or screens almost no nnz:
        use normal sparse-residual-active path
    else:
        run active CCD inside C
        run full reconciliation/KKT against all coordinates
        add violators
        repeat if needed
```

This makes the strong set the outer candidate universe while the active set
remains the inner fast set.

Expected benefit:

- possibly fewer full-coordinate reconciliation sweeps;
- fewer zero-coordinate scans before the active set stabilizes;
- better early candidates for active-only CCD;
- unlikely to give another 2x by itself.

Concrete retest:

```text
sparse-residual-active
vs
sparse-residual-active + historical strong candidate sets
```

Required metrics:

- total elapsed;
- selected variance;
- prediction RMSE and max abs diff versus active baseline;
- objective diff versus active baseline;
- max KKT/subgradient residual;
- active epochs;
- full reconciliation sweeps;
- strong-set size and nnz;
- screened nnz;
- KKT violator count and nnz;
- gradient/Hessian nnz scanned;
- update nnz scanned;
- reconciliation/KKT nnz scanned;
- fallback-to-full count.

Success bar:

```text
>=5-10% total-time improvement over sparse-residual-active
with no worse selection drift than the active baseline
```

### GAP-safe after active sets

Active sets do not fix the main GAP-safe failure. Basic GAP-safe screened many
columns but almost no nnz:

```text
13.8-15.3% of eligible zero columns
0.014-0.026% of eligible nnz
```

Since the hot path is sparse `fitted[row]` gathers, rare-column screening does
not buy much.

Where active sets could still help:

1. run GAP-safe only during strict polish / decision-sensitive candidates;
2. use late active-set iterates, where the dual point may be better;
3. use GAP-safe to skip some full reconciliation/KKT checks, not active CCD;
4. combine with dual extrapolation only if the critical-radius diagnostic says
   hot columns are near-screenable.

The decisive retest is diagnostic-only:

```text
Run GAP-safe at active strict-polish checkpoints.
Use centered/fold-weighted norms.
Report critical-radius curves by nnz bucket.
```

Required table:

```text
radius      implied gap shrink     screened nnz share     hot-nnz screened
R           1x                     ...
R/2         4x smaller gap          ...
R/4         16x smaller gap         ...
R/8         64x smaller gap         ...
R/16        256x smaller gap        ...
R/32        1024x smaller gap       ...
R=0         oracle center           ...
```

Interpretation:

- if `R/2` or `R/4` screens more than 5-10% of eligible nnz, celer-style dual
  extrapolation is worth a small prototype;
- if meaningful hot-nnz screening only appears at `R/16` or smaller, close
  GAP-safe again.

### Fold-local/layout after active sets

Fold-local compact CSC should now be measured against the active baseline:

```text
sparse-residual-active
vs
fold-local sparse-residual-active
```

not against the old plain fitted-response baseline.

The active path reduces the amount of full-column work, so the possible layout
gain is smaller in active-only epochs. It can still matter for:

- initial full sweeps;
- full reconciliation sweeps;
- strict polish;
- final full-data fit;
- hot active coordinates with many rows.

Measure:

- total wall time;
- build time;
- solve time;
- peak RSS;
- gradient/Hessian time;
- update time;
- reconciliation time;
- cycles per scanned nnz;
- L1D/LLC/dTLB misses per scanned nnz;
- active-only epochs versus full sweeps.

The success bar is total-time improvement over `sparse-residual-active`, not
only cleaner setup or lower isolated kernel time.

### IRLS/WLS/Gram after active sets

All-feature WLS/IRLS remains rejected. Active sets make only a narrow late-path
accelerator more plausible.

Possible guard:

```text
if active_set_size <= 300 or 500
and active set has been stable for N sweeps
and active-only CCD is still doing many epochs:

    build active-only WLS/Gram
    run a few dense/CD/Newton polish steps
    update beta/fitted exactly
    return to fitted-response active CCD
    full reconciliation/KKT before acceptance
```

The key amortization question is:

```text
Gram/WLS build time
vs
active CCD epochs saved
```

This is more plausible than all-feature WLS, but still lower priority than
validating the active path and retesting strong rules on top of it.

### Retest priority after active sets

| Idea | How active-set lessons apply | Retest priority |
|---|---|---:|
| Historical strong rules | Use as outer candidate-set initializer; active set remains inner solver | High |
| GAP-safe | Use only as late-polish/reconciliation diagnostic unless nnz screening improves | Low-medium |
| Compact CSC/layout | Retest against active baseline; target full/reconciliation sweeps and hot active columns | High |
| CSR row-wise KKT | Only if reconciliation/KKT becomes bottleneck | Low |
| Active WLS/Gram | Use only when active set is small/stable and many active epochs remain | Medium |
| Celer-style solver | Separate larger solver project | Low for current PR |

Do not immediately retest these just because active sets worked:

- basic GAP-safe production mask;
- duplicate compression;
- AVX2 gather;
- unroll/prefetch variants;
- all-feature WLS/IRLS;
- row-pass full-gradient replacement.

## Questions For External Review

1. Is the lazy selective-polish strategy acceptable as a Cyclops CV contract if
   all decision-sensitive candidates and the selected final model are strictly
   polished?

2. Should active-set convergence be considered exact with a full reconciliation
   sweep, or should acceptance require an explicit KKT/subgradient threshold?

3. Is there a better way to use active sets without changing auto-CV decisions?

4. Given the current hot loop, is a data-layout redesign more promising than
   more screening?

5. Is there a celer-style dual extrapolation method that could screen hot nnz on
   this PLP matrix distribution, despite basic GAP-safe screening only rare
   columns?

6. Can a narrow active-set WLS/Gram warm accelerator be made cheap enough to help
   after active sets are already small?

7. Should Cyclops expose this as an opt-in `sparse-residual-active` kernel first,
   rather than changing defaults?

8. What correctness metrics should be required for accepting small differences
   in selected variance/support when prediction RMSE and objective differences
   are tiny?

## Suggested Next Step

If the goal is a near-term maintainer-facing PR:

```text
Do not pursue another speculative kernel first.
Split and clean the current fitted-response + active/lazy-polish work.
Run multi-dataset PLP validation.
Present active mode as opt-in until semantics are reviewed.
```

If the goal is another research cycle:

```text
Focus on:
    1. sparse-residual-active + historical strong candidate sets,
    2. diagnostic-only GAP-safe at active strict-polish checkpoints,
    3. fold-local/layout retest against sparse-residual-active,
    4. narrow active-set WLS/Gram accelerator only if active epochs remain high.

Do not spend more time on:
    basic GAP-safe,
    simple strong-rule threshold tweaks,
    row-wise KKT scanner,
    duplicate compression,
    scalar unroll/prefetch/guard micro-patches.
```

## Retest Results After Active Sets

Date: 2026-06-13.

Raw benchmark artifacts:

```text
docs/perf-results/2026-06-13-active-retests/
```

Benchmark setup:

```text
Data:
    /home/egill/github/Cyclops/extras/benchmarks/activeSetLogistic/results/
        ducky-dementia-plp-sample-50000.rds
        ducky-dementia-plp-sample-200000.rds

Mode:
    exact CV, folds=3, threads=3

Control:
    sparseConvergenceType=fitted-change
    cvPolishPolicy=selective
    cvPolishTolerance=1e-8
    cvPolishMaxIterations=1500
    cvPolishInitialCandidates=1
    cvPolishStopProposal=TRUE

Noise control:
    one foreground benchmark process at a time
    no single-thread runs
```

These are single-run measurements, so small differences should be treated as
directional until repeated on an otherwise idle machine. Larger differences
are strong enough to drive the next implementation choice.

### Active plus historical strong candidate sets

Implemented a new combined kernel:

```text
sparse-residual-active-strong
```

This uses the strong-rule wrapper as the outer candidate-set mechanism, but
uses `sparse-residual-active` for the restricted/full solves inside the strong
wrapper. This tests the relevant question:

```text
sparse-residual-active
vs
sparse-residual-active + strong candidate sets
```

instead of comparing plain strong rules against the non-active baseline.

Results:

| Sample | Kernel | Time | Variance | Nonzero | KKT max | Notes |
|---:|---|---:|---:|---:|---:|---|
| 50k | `sparse-residual-active` | 3.670s | 0.02648248 | 135 | 0.01795 | baseline |
| 50k | `sparse-residual-active-strong` | 4.001s | 0.02648160 | 135 | 0.00484 | slower, tiny prediction drift |
| 200k | `sparse-residual-active` | 30.211s | 0.01342480 | 263 | 0.01604 | baseline |
| 200k | `sparse-residual-active-strong` | 28.243s | 0.01342480 | 260 | 0.08713 | 1.07x faster, tiny prediction drift |

200k counters:

```text
sparse-residual-active:
    ccdEpochs: 8070
    activeSetOnlyEpochs: 7895
    activeSetFullReconciliationSweeps: 58

sparse-residual-active-strong:
    ccdEpochs: 6488
    activeSetOnlyEpochs: 6436
    activeSetFullReconciliationSweeps: 73
    fullKktScans: 27
    strongRuleScreenedNnz: 285,457,760
    strongRuleFullSetBypasses: 7
    nnzKkt: 584,983,353
```

Interpretation:

```text
Strong rules on top of active sets can reduce CCD epochs, but they also add
full-KKT/reconciliation work. On 200k without fold-local layout this gave only
a modest net win. On 50k it was slower.
```

This is not yet a clear production win. It is more promising as an optional
candidate-set initializer than as the next main PR.

### GAP-safe diagnostic at active checkpoints

Ran diagnostic-only GAP-safe radius tracing on 200k active CV:

| Kernel | Time | Variance | Nonzero | GAP-safe checks | GAP-safe seconds |
|---|---:|---:|---:|---:|---:|
| `sparse-residual-active` + diagnostic | 44.622s | 0.01342480 | 263 | 20 | 35.90s |

Compared with the 200k active baseline at 30.211s, the diagnostic adds about
14.4s wall time in this run. The internal timed GAP-safe work sums to 35.9s
across fold/lambda checks because fold work is parallelized over the three CV
threads.

This active-checkpoint result differs from the earlier basic GAP-safe result:
at some active-polished checkpoints the safe radius is much smaller, and the
current radius can screen real nnz. Aggregated across 20 checks:

```text
current uncentered radius:
    screened columns: 192,151 / 326,327 = 58.9%
    screened nnz:      87,292,063 / 502,395,824 = 17.4%

current centered radius:
    screened columns: 192,302 / 326,327 = 58.9%
    screened nnz:      89,107,985 / 502,395,824 = 17.7%
```

Median screening share by radius divisor:

| Radius | Implied gap factor | Median nnz screened | Median hot nnz screened | Centered median nnz screened | Centered median hot nnz screened |
|---:|---:|---:|---:|---:|---:|
| R | 1 | 3.6% | 0.0% | 3.6% | 0.0% |
| R/2 | 1/4 | 10.2% | 0.0% | 10.4% | 0.0% |
| R/4 | 1/16 | 22.9% | 6.3% | 23.5% | 7.1% |
| R/8 | 1/64 | 48.3% | 36.7% | 55.9% | 46.7% |
| R/16 | 1/256 | 77.5% | 72.3% | 80.6% | 76.0% |
| R/32 | 1/1024 | 89.7% | 87.2% | 91.9% | 89.9% |
| R/64 | 1/4096 | 95.4% | 94.3% | 96.1% | 95.2% |

Last-check details:

```text
duality gap: 0.258
radius: 0.359
eligible nnz: 36,117,887
eligible hot nnz: 30,442,858

At R:
    screened nnz: 1,363,783 to 1,371,012
    screened hot nnz: 0

At R/4:
    screened nnz: 8,348,174 to 8,659,063
    screened hot nnz: 3,173,230 to 3,470,786

At R/8:
    screened nnz: 17,477,286 to 20,182,846
    screened hot nnz: 11,948,029 to 14,653,589
```

Interpretation:

```text
Active sets do make GAP-safe more interesting than the earlier basic diagnostic
suggested. However, the current diagnostic implementation is much too expensive
to wire in directly. The useful production version would need to be dynamic,
cheap, and probably extrapolated, and it would need to avoid spending a full
screening pass unless it can skip enough future reconciliation work.
```

Updated decision:

```text
Do not implement a production GAP-safe inactive mask yet.
Keep GAP-safe as a diagnostic/late-polish research path.
If revisited, focus on:
    dynamic checkpoints,
    dual extrapolation,
    centered norms,
    and nnz-weighted triggering.
```

### Fold-local compact CV against active baseline

Ran `sparse-residual-active` with:

```text
CYCLOPS_FOLD_LOCAL_CV=1
```

Results:

| Sample | Kernel/layout | Time | Variance | Nonzero | KKT max | Fold-local build time |
|---:|---|---:|---:|---:|---:|---:|
| 50k | current active | 3.670s | 0.02648248 | 135 | 0.01795 | 0 |
| 50k | fold-local active | 3.048s | 0.02648123 | 135 | 0.01795 | 0.197s |
| 200k | current active | 30.211s | 0.01342480 | 263 | 0.01604 | 0 |
| 200k | fold-local active | 21.404s | 0.01342480 | 263 | 0.01604 | 0.771s |

Fold-local plus active+strong on 200k:

| Kernel/layout | Time | Variance | Nonzero | KKT max |
|---|---:|---:|---:|---:|
| fold-local `sparse-residual-active` | 22.568s | 0.01342480 | 263 | 0.01604 |
| fold-local `sparse-residual-active-strong` | 25.221s | 0.01342480 | 260 | 0.08713 |

The fold-local active baseline differs slightly between the separate run
(`21.404s`) and same-process strong comparison (`22.568s`), consistent with
normal run-to-run noise. Both are much faster than the non-fold-local active
baseline.

Interpretation:

```text
Fold-local compact CV is the strongest measured new improvement in this round.
It improves 50k by about 1.20x and 200k by about 1.35-1.41x on top of the
active fitted-response kernel, while preserving selected variance, support
size, and KKT counters in these runs.

Active+strong does not stack with fold-local in this benchmark; it becomes
slower because the remaining zero-coordinate/reconciliation work is already
cheaper and the strong wrapper still adds KKT scans and bookkeeping.
```

### Updated Recommendation

Current priority:

```text
1. Treat fold-local compact CV as the next main implementation path.
2. Keep sparse-residual-active as the baseline solver.
3. Keep active+strong as experimental; do not merge it into the main path yet.
4. Keep GAP-safe as diagnostic/research; do not wire in the current full pass.
```

Specific follow-up work:

```text
Fold-local:
    - inspect whether the current C++ fold-local path can be simplified; partly done with driver cleanup;
    - add stronger tests for exact prediction/objective equivalence; done with fold-local CV equivalence test;
    - measure peak memory and build-time scaling on 500k/1M/2M samples;
    - profile fold-local active with perf against current active; done on 200k pinned run.

Strong:
    - investigate why KKT max differs for active+strong;
    - only continue if the correctness story is clean;
    - retest after fold-local code is stable.

GAP-safe:
    - do not productionize the existing pass;
    - if revisited, prototype cheap dynamic/extrapolated diagnostics only;
    - trigger by expected nnz savings, not column count.
```

## Fold-Local Perf Profile

Date: 2026-06-16.

Raw artifacts:

```text
docs/perf-results/2026-06-16-foldlocal-profile/
```

Setup:

```text
200k PLP sample
exact CV, folds=3, threads=3
taskset -c 0-2
perf stat
kernel=sparse-residual-active
CYCLOPS_FOLD_LOCAL_CV=1 for fold-local run
```

Results:

| Variant | Time | Variance | Nonzero | KKT max |
|---|---:|---:|---:|---:|
| current active | 27.805s | 0.0134248 | 263 | 0.01604303 |
| fold-local active | 21.131s | 0.0134248 | 263 | 0.01604303 |

Fold-local speedup in the pinned perf run:

```text
elapsed:       1.316x
core cycles:   -29.1%
instructions:  -31.5%
cache misses:  -29.9%
L1D misses:    -42.0%
dTLB misses:   -22.6%
```

Sampled perf report:

| Symbol | Current active | Fold-local active |
|---|---:|---:|
| `updateSparseLogisticState` | 38.44% | 38.33% |
| `sparseLogisticAvx2IndicatorGradientHessianDouble` | 33.40% | 32.75% |
| `ensureSparseLogisticWeightedColumnViews` | 3.42% | below 1% |
| `computeSparseLogisticGradientAndHessian` wrapper | 2.84% | 2.91% |
| `getLogLikelihood` | 1.93% | 1.20% |
 
The sampled profile confirms that fold-local does not change the dominant
kernel mix. The same state-update and indicator GH loops are hot; fold-local
wins by reducing total work, weighted-view overhead, and cache/TLB pressure.

Fold-local build cost:

```text
foldLocalBuildAttempts: 3
foldLocalBuildSuccesses: 3
foldLocalBuildSeconds: 0.7686
foldLocalDataBuildSeconds: 0.7575
```

Interpretation:

```text
Fold-local is not just a wall-time artifact. Perf shows lower instruction
count, fewer core cycles, and substantially fewer L1D misses. The compact
fold data reduces both work and locality penalties, and the native build cost
is small relative to the solve-time gain on 200k.
```
