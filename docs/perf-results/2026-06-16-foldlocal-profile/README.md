## 2026-06-16 Fold-Local Active Perf Profile

Purpose:

```text
Profile current sparse-residual-active CV against fold-local sparse-residual-active CV.
```

Benchmark setup:

```text
data: /home/egill/github/Cyclops/extras/benchmarks/activeSetLogistic/results/ducky-dementia-plp-sample-200000.rds
mode: cv
folds: 3
threads: 3
CPU affinity: taskset -c 0-2
kernel: sparse-residual-active
sparseConvergenceType: fitted-change
cvPolishPolicy: selective
cvPolishTolerance: 1e-8
cvPolishMaxIterations: 1500
cvPolishInitialCandidates: 1
cvPolishStopProposal: TRUE
```

Artifacts:

```text
cyclops-200k-active.csv
cyclops-200k-foldlocal-active.csv
perf-stat-200k-active.csv
perf-stat-200k-foldlocal-active.csv
cyclops-200k-active-record.csv
cyclops-200k-foldlocal-active-record.csv
perf-record-200k-active.data
perf-record-200k-foldlocal-active.data
perf-report-200k-active.txt
perf-report-200k-foldlocal-active.txt
```

Elapsed results:

```text
current active:    27.805s
fold-local active: 21.131s
speedup:           1.316x
```

Correctness/equivalence counters:

```text
selected variance:
    current active:    0.0134248
    fold-local active: 0.0134248

nonzero:
    current active:    263
    fold-local active: 263

kkt_max_violation:
    current active:    0.01604303
    fold-local active: 0.01604303
```

Solver counters:

```text
current active:
    ccdEpochs: 8070
    activeSetOnlyEpochs: 7895
    activeSetFullReconciliationSweeps: 58
    cvScoringSeconds: 0.1417

fold-local active:
    ccdEpochs: 5429
    activeSetOnlyEpochs: 5289
    activeSetFullReconciliationSweeps: 50
    foldLocalBuildAttempts: 3
    foldLocalBuildSuccesses: 3
    foldLocalBuildSeconds: 0.7686
    foldLocalDataBuildSeconds: 0.7575
    foldLocalCcdSetupSeconds: 0.0111
    foldLocalInitSeconds: 0.0097
    cvBetaSyncSeconds: 0.7068
    cvScoringSeconds: 0.0391
```

Perf counter deltas, using `cpu_core/*` events:

| Counter | Current active | Fold-local active | Fold-local / current | Reduction |
|---|---:|---:|---:|---:|
| cycles | 356,055,640,470 | 252,370,008,991 | 0.709 | 29.1% |
| instructions | 1,268,403,088,704 | 869,457,645,243 | 0.685 | 31.5% |
| cache references | 4,837,304,226 | 3,103,050,550 | 0.641 | 35.9% |
| cache misses | 2,364,512,812 | 1,658,418,884 | 0.701 | 29.9% |
| L1D loads | 218,362,721,765 | 144,379,791,181 | 0.661 | 33.9% |
| L1D load misses | 41,248,682,301 | 23,929,780,711 | 0.580 | 42.0% |
| dTLB loads | 218,362,721,765 | 144,379,791,181 | 0.661 | 33.9% |
| dTLB load misses | 21,094,139 | 16,333,201 | 0.774 | 22.6% |

Interpretation:

```text
Fold-local CV is faster because it reduces the amount of coordinate-descent
work and improves locality. In this pinned 200k run it cuts elapsed time by
about 24%, core cycles by about 29%, instructions by about 31%, and L1D load
misses by about 42%, while preserving selected variance, support size, and the
reported KKT counter.

The native fold-local build cost is small relative to solve savings:
0.77s total build time for three training folds.
```

Sampled perf profiles:

```text
current active record:
    elapsed: 28.912s
    samples: 7,817
    event count: 356,347,596,528 cpu_core/cycles/P

fold-local active record:
    elapsed: 20.878s
    samples: 5,376
    event count: 251,585,648,740 cpu_core/cycles/P
```

Top sampled user-space symbols:

| Symbol | Current active | Fold-local active |
|---|---:|---:|
| `updateSparseLogisticState` | 38.44% | 38.33% |
| `sparseLogisticAvx2IndicatorGradientHessianDouble` | 33.40% | 32.75% |
| `ensureSparseLogisticWeightedColumnViews` | 3.42% | not above 1% |
| `computeSparseLogisticGradientAndHessian` wrapper | 2.84% | 2.91% |
| `getLogLikelihood` | 1.93% | 1.20% |

Profile interpretation:

```text
The dominant hot-loop mix is essentially unchanged: state updates plus the
AVX2 indicator gradient/Hessian loop still dominate. Fold-local wins by doing
less total work and by improving memory behavior, not by moving execution into
a different kernel.

The weighted-column-view setup symbol appears in the current active profile
but drops below the 1% report threshold in the fold-local profile, consistent
with compact fold data removing part of the view/materialization overhead.
```
