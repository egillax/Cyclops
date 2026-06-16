## 2026-06-13 Active Retests

Benchmark setup:

```text
mode=cv
folds=3
threads=3
sparseConvergenceType=fitted-change
cvPolishPolicy=selective
cvPolishTolerance=1e-8
cvPolishMaxIterations=1500
cvPolishInitialCandidates=1
cvPolishStopProposal=TRUE
```

Data files:

```text
/home/egill/github/Cyclops/extras/benchmarks/activeSetLogistic/results/ducky-dementia-plp-sample-50000.rds
/home/egill/github/Cyclops/extras/benchmarks/activeSetLogistic/results/ducky-dementia-plp-sample-200000.rds
```

Artifacts:

```text
cyclops-50k-active-strong-retest.csv
cyclops-200k-active-strong-retest.csv
cyclops-200k-active-gapsafe.csv
cyclops-gap-active-radius-200k.csv
cyclops-50k-active-foldlocal.csv
cyclops-200k-active-foldlocal.csv
cyclops-200k-foldlocal-active-strong.csv
```

Summary:

```text
50k active vs active+strong:
    3.670s vs 4.001s

200k active vs active+strong:
    30.211s vs 28.243s

200k active + GAP-safe diagnostic:
    44.622s total
    35.90s GAP-safe diagnostic time

50k current active vs fold-local active:
    3.670s vs 3.048s

200k current active vs fold-local active:
    30.211s vs 21.404s

200k fold-local active vs fold-local active+strong:
    22.568s vs 25.221s
```
