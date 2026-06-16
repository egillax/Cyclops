#!/usr/bin/env Rscript

configure_andromeda <- function() {
  andromeda_temp <- path.expand(Sys.getenv("ANDROMEDA_TEMP", "~/andromeda_tmp"))
  dir.create(andromeda_temp, recursive = TRUE, showWarnings = FALSE)

  options(andromedaMemoryLimit = as.numeric(Sys.getenv("ANDROMEDA_MEMORY_GB", "40")))
  options(andromedaTempFolder = andromeda_temp)
  options(andromedaThreads = as.integer(Sys.getenv("ANDROMEDA_THREADS", "8")))
}

usage <- function(status = 0) {
  cat(
    "Usage:\n",
    "  Rscript extras/benchmarks/benchmark_plp_logistic_kernels.R \\\n",
    "    --matrix-rds data.rds --out results.csv\n\n",
    "  Rscript extras/benchmarks/benchmark_plp_logistic_kernels.R \\\n",
    "    --plp-data /path/to/plpData --outcome-id 123 --out results.csv\n\n",
    "Required input, choose one:\n",
    "  --matrix-rds PATH        RDS list with x=dgCMatrix and y=0/1 vector.\n",
    "  --plp-data PATH          Saved PLP data directory or RDS object.\n\n",
    "Common options:\n",
    "  --out PATH               CSV output path. Default: kernel-benchmark-results.csv\n",
    "  --reference-matrix-rds PATH  Original matrix for compressed-column diagnostics.\n",
    "  --kernels LIST           Comma-separated kernels. Default: legacy,sparse-residual\n",
    "  --mode MODE              fixed or cv. Default: cv\n",
    "  --variance VALUE         Fixed-fit Laplace variance. Default: 0.02\n",
    "  --folds N                CV folds. Default: 3\n",
    "  --threads N              Cyclops threads. Default: folds\n",
    "  --seed N                 CV seed. Default: 42\n",
    "  --tolerance VALUE        Cyclops tolerance. Default: 1e-6\n",
    "  --convergence-type TYPE  Cyclops convergence type. Default: gradient\n",
    "  --sparse-convergence-type TYPE  auto, fitted-change, or coordinate-impact. Default: auto\n",
    "  --sparse-active-kkt-tolerance VALUE  Active-coordinate KKT gate; negative disables. Default: -1\n",
    "  --sparse-active-kkt-movement-tolerance VALUE  Movement threshold before KKT checks. Default: -1\n",
    "  --sparse-active-kkt-polish-tolerance VALUE  KKT gate used by polish solves. Default: -1\n",
    "  --sparse-active-kkt-check-every N  Check period after movement threshold. Default: 1\n",
    "  --cv-polish-policy POLICY  none, selective, or all. Default: none\n",
    "  --cv-polish-tolerance VALUE  Stricter CV/final-fit tolerance. Default: -1\n",
    "  --cv-polish-max-iterations N  Polish iteration limit. Default: -1\n",
    "  --cv-polish-initial-candidates N  Initial auto-search candidates to polish. Default: 1\n",
    "  --cv-polish-stop-proposal TRUE/FALSE  Polish final stop proposal. Default: TRUE\n",
    "  --max-iterations N       Max CCD iterations. Default: 1000\n",
    "  --lower-limit VALUE      CV lower variance bound. Default: 0.01\n",
    "  --upper-limit VALUE      CV upper variance bound. Default: 20\n",
    "  --grid-steps N           Grid steps for grid CV. Default: 10\n",
    "  --cv-type TYPE           auto or grid. Default: auto\n",
    "  --repetitions N          Refit each kernel N times. Default: 1\n",
    "  --sample-size N          Optional row sample after loading.\n",
    "  --outcome-id ID          Required for PLP data without embedded population.\n",
    "  --help                   Show this help.\n",
    sep = ""
  )
  quit(save = "no", status = status)
}

parse_args <- function(argv) {
  args <- list(
    matrix_rds = NA_character_,
    plp_data = NA_character_,
    out = "kernel-benchmark-results.csv",
    reference_matrix_rds = NA_character_,
    kernels = "legacy,sparse-residual",
    mode = "cv",
    variance = 0.02,
    folds = 3L,
    threads = NA_integer_,
    seed = 42L,
    tolerance = 1e-6,
    convergence_type = "gradient",
    sparse_convergence_type = "auto",
    sparse_active_kkt_tolerance = -1,
    sparse_active_kkt_movement_tolerance = -1,
    sparse_active_kkt_polish_tolerance = -1,
    sparse_active_kkt_check_every = 1L,
    cv_polish_policy = "none",
    cv_polish_tolerance = -1,
    cv_polish_max_iterations = -1L,
    cv_polish_initial_candidates = 1L,
    cv_polish_stop_proposal = TRUE,
    max_iterations = 1000L,
    lower_limit = 0.01,
    upper_limit = 20,
    grid_steps = 10L,
    cv_type = "auto",
    repetitions = 1L,
    sample_size = NA_integer_,
    outcome_id = NA_integer_
  )

  i <- 1L
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (key %in% c("--help", "-h")) {
      usage(0)
    }
    if (!startsWith(key, "--") || i == length(argv)) {
      stop("Expected --key value argument near: ", key)
    }
    value <- argv[[i + 1L]]
    name <- gsub("-", "_", substring(key, 3L))
    if (!name %in% names(args)) {
      stop("Unknown argument: ", key)
    }
    args[[name]] <- value
    i <- i + 2L
  }

  integer_args <- c("folds", "threads", "seed", "max_iterations", "grid_steps",
                    "repetitions", "sample_size", "outcome_id",
                    "sparse_active_kkt_check_every", "cv_polish_max_iterations",
                    "cv_polish_initial_candidates")
  numeric_args <- c("variance", "tolerance", "lower_limit", "upper_limit",
                    "sparse_active_kkt_tolerance",
                    "sparse_active_kkt_movement_tolerance",
                    "sparse_active_kkt_polish_tolerance", "cv_polish_tolerance")
  for (name in integer_args) {
    args[[name]] <- if (is.na(args[[name]])) NA_integer_ else as.integer(args[[name]])
  }
  for (name in numeric_args) {
    args[[name]] <- as.numeric(args[[name]])
  }
  if (is.character(args$cv_polish_stop_proposal)) {
    args$cv_polish_stop_proposal <- tolower(args$cv_polish_stop_proposal) %in%
      c("1", "true", "t", "yes", "y")
  }
  if (!args$convergence_type %in% c("gradient", "lange", "mittal", "zhang")) {
    stop("--convergence-type must be gradient, lange, mittal, or zhang")
  }
  if (!args$sparse_convergence_type %in% c("auto", "fitted-change", "coordinate-impact")) {
    stop("--sparse-convergence-type must be auto, fitted-change, or coordinate-impact")
  }
  if (args$sparse_active_kkt_check_every < 1L) {
    stop("--sparse-active-kkt-check-every must be at least 1")
  }
  if (!args$cv_polish_policy %in% c("none", "selective", "all")) {
    stop("--cv-polish-policy must be none, selective, or all")
  }
  if (args$cv_polish_initial_candidates < 0L) {
    stop("--cv-polish-initial-candidates must be non-negative")
  }
  args
}

require_package <- function(name, optional = FALSE) {
  if (!requireNamespace(name, quietly = TRUE)) {
    if (optional) {
      return(FALSE)
    }
    stop("Package '", name, "' is required")
  }
  TRUE
}

as_local_df <- function(x) {
  if (inherits(x, "tbl_sql") || inherits(x, "tbl_dbi") || inherits(x, "tbl_lazy")) {
    require_package("dplyr")
    return(dplyr::collect(x))
  }
  as.data.frame(x)
}

load_rds_input <- function(path) {
  obj <- readRDS(path)
  if (is.list(obj) && all(c("x", "y") %in% names(obj))) {
    require_package("Matrix")
    x <- obj$x
    if (!inherits(x, "dgCMatrix")) {
      x <- Matrix::Matrix(x, sparse = TRUE)
      x <- as(x, "dgCMatrix")
    }
    y <- as.integer(obj$y)
    return(list(x = x, y = y, metadata = obj$metadata, source = "matrix-rds"))
  }
  obj
}

load_reference_matrix <- function(path, expected_y) {
  if (is.na(path)) {
    return(NULL)
  }
  obj <- load_rds_input(path)
  if (!is.list(obj) || !all(c("x", "y") %in% names(obj))) {
    stop("--reference-matrix-rds must contain list entries x and y")
  }
  if (length(obj$y) != length(expected_y) || any(as.integer(obj$y) != as.integer(expected_y))) {
    stop("--reference-matrix-rds y does not match the fitted data")
  }
  obj
}

load_plp_input <- function(path, outcome_id, sample_size, seed) {
  require_package("dplyr")
  require_package("Andromeda")
  require_package("Cyclops")
  require_package("PatientLevelPrediction")

  configure_andromeda()

  plp_data <- if (dir.exists(path)) {
    PatientLevelPrediction::loadPlpData(path)
  } else {
    load_rds_input(path)
  }
  if (is.list(plp_data) && all(c("x", "y") %in% names(plp_data))) {
    return(plp_data)
  }

  population <- NULL
  if ("population" %in% names(plp_data)) {
    population <- as_local_df(plp_data$population)
  } else if (!is.na(outcome_id)) {
    settings <- PatientLevelPrediction::createStudyPopulationSettings(
      requireTimeAtRisk = FALSE
    )
    population <- PatientLevelPrediction::createStudyPopulation(
      plpData = plp_data,
      outcomeId = outcome_id,
      populationSettings = settings
    )
    population <- as_local_df(population)
  } else {
    stop("PLP input needs an embedded population or --outcome-id")
  }

  if (!"rowId" %in% names(population)) {
    stop("Population must contain rowId")
  }
  if ("outcomeCount" %in% names(population)) {
    y <- as.integer(population$outcomeCount > 0)
  } else if ("y" %in% names(population)) {
    y <- as.integer(population$y)
  } else {
    stop("Population must contain outcomeCount or y")
  }
  if (!is.na(sample_size) && sample_size < length(y)) {
    set.seed(seed)
    rows <- sort(sample(seq_along(y), sample_size))
    population <- population[rows, , drop = FALSE]
    y <- y[rows]
  }

  outcomes <- data.frame(rowId = population$rowId, y = y)
  outcomes <- outcomes[order(outcomes$rowId), , drop = FALSE]
  y <- outcomes$y
  andromeda <- Andromeda::andromeda(outcomes = outcomes)
  covariates <- plp_data$covariateData$covariates |>
    dplyr::inner_join(
      andromeda$outcomes |> dplyr::select(rowId),
      by = "rowId",
      copy = TRUE
    ) |>
    dplyr::select(rowId, covariateId, covariateValue)
  nnz <- tryCatch(
    covariates |>
      dplyr::summarise(nnz = dplyr::n()) |>
      dplyr::pull(nnz),
    error = function(e) NA_integer_
  )

  cyclops_data <- Cyclops::convertToCyclopsData(
    outcomes = andromeda$outcomes,
    covariates = covariates,
    modelType = "lr",
    addIntercept = TRUE,
    checkRowIds = FALSE,
    quiet = TRUE
  )

  list(
    cyclops_data = cyclops_data,
    x = NULL,
    y = y,
    rows = length(y),
    features = Cyclops::getNumberOfCovariates(cyclops_data) - 1L,
    nnz = as.numeric(nnz)[1L],
    source = "plp-data",
    andromeda = andromeda
  )
}

sample_rows <- function(data, n, seed) {
  if (is.na(n) || n >= length(data$y)) {
    return(data)
  }
  set.seed(seed)
  rows <- sort(sample(seq_along(data$y), n))
  data$x <- data$x[rows, , drop = FALSE]
  data$y <- data$y[rows]
  data$rows <- nrow(data$x)
  data$features <- ncol(data$x)
  data$nnz <- length(data$x@x)
  data
}

make_cyclops_data <- function(x, y) {
  require_package("Matrix")
  require_package("Cyclops")
  x <- as(x, "dgCMatrix")
  triplet <- Matrix::summary(x)
  outcomes <- data.frame(rowId = seq_along(y), y = as.integer(y))
  covariates <- data.frame(
    rowId = as.integer(triplet$i),
    covariateId = as.integer(triplet$j),
    covariateValue = as.numeric(triplet$x)
  )
  Cyclops::convertToCyclopsData(
    outcomes = outcomes,
    covariates = covariates,
    modelType = "lr",
    addIntercept = TRUE
  )
}

stable_log1pexp <- function(eta) {
  ifelse(eta > 0, eta + log1p(exp(-eta)), log1p(exp(eta)))
}

compute_metrics <- function(x, y, beta, variance) {
  require_package("Matrix")
  if (is.null(beta) || all(is.na(beta))) {
    return(list(
      log_likelihood_check = NA_real_,
      objective_check = NA_real_,
      kkt_max_violation = NA_real_,
      prediction = NA_real_
    ))
  }
  if (is.null(x)) {
    return(list(
      log_likelihood_check = NA_real_,
      objective_check = NA_real_,
      kkt_max_violation = NA_real_,
      prediction = NA_real_
    ))
  }

  beta <- as.numeric(beta)
  beta[is.na(beta)] <- 0
  if (length(beta) == ncol(x) + 1L) {
    eta <- beta[1L] + as.numeric(x %*% beta[-1L])
  } else if (length(beta) == ncol(x)) {
    eta <- as.numeric(x %*% beta)
  } else {
    return(list(
      log_likelihood_check = NA_real_,
      objective_check = NA_real_,
      kkt_max_violation = NA_real_,
      prediction = NA_real_
    ))
  }
  log_likelihood <- sum(y * eta - stable_log1pexp(eta))
  lambda <- sqrt(2 / variance)
  penalized_beta <- if (length(beta) == ncol(x) + 1L) beta[-1L] else beta
  objective <- log_likelihood - lambda * sum(abs(penalized_beta))
  p <- stats::plogis(eta)
  residual <- y - p
  gradient <- as.numeric(Matrix::crossprod(x, residual))
  intercept_violation <- abs(sum(residual))
  active <- abs(penalized_beta) > 0
  coefficient_violation <- numeric(length(penalized_beta))
  coefficient_violation[active] <- abs(gradient[active] - lambda * sign(penalized_beta[active]))
  coefficient_violation[!active] <- pmax(0, abs(gradient[!active]) - lambda)
  active_violation <- if (any(active)) {
    max(coefficient_violation[active], na.rm = TRUE)
  } else {
    0
  }
  inactive_violation <- if (any(!active)) {
    max(coefficient_violation[!active], na.rm = TRUE)
  } else {
    0
  }
  list(
    log_likelihood_check = log_likelihood,
    objective_check = objective,
    kkt_max_violation = max(intercept_violation, coefficient_violation, na.rm = TRUE),
    kkt_intercept_violation = intercept_violation,
    kkt_active_violation = active_violation,
    kkt_inactive_violation = inactive_violation,
    prediction = p
  )
}

expand_duplicate_beta <- function(beta, data, reference) {
  compression <- data$metadata$duplicateCompression
  if (is.null(compression) || is.null(reference)) {
    return(beta)
  }

  beta <- as.numeric(beta)
  beta[is.na(beta)] <- 0
  has_intercept <- length(beta) == ncol(data$x) + 1L
  penalized <- if (has_intercept) beta[-1L] else beta
  if (length(penalized) != ncol(data$x)) {
    stop("Compressed beta length does not match compressed matrix")
  }

  kept <- as.integer(compression$keptColumns)
  original_columns <- as.integer(compression$originalColumns)
  if (length(kept) != ncol(data$x) || original_columns != ncol(reference$x)) {
    stop("Duplicate-compression metadata does not match compressed/reference matrices")
  }

  expanded <- numeric(original_columns)
  expanded[kept] <- penalized
  if (has_intercept) {
    c(beta[[1L]], expanded)
  } else {
    expanded
  }
}

compute_prediction_metrics <- function(fit, y, beta, variance) {
  if (is.null(beta) || all(is.na(beta))) {
    return(list(
      log_likelihood_check = NA_real_,
      objective_check = NA_real_,
      prediction = NA_real_
    ))
  }
  beta <- as.numeric(beta)
  beta[is.na(beta)] <- 0
  prediction <- tryCatch(
    as.numeric(stats::predict(fit)),
    error = function(e) NA_real_
  )
  if (length(prediction) != length(y) || any(is.na(prediction))) {
    return(list(
      log_likelihood_check = NA_real_,
      objective_check = NA_real_,
      prediction = NA_real_
    ))
  }

  eps <- .Machine$double.eps
  prediction <- pmin(pmax(prediction, eps), 1 - eps)
  log_likelihood <- sum(y * log(prediction) + (1 - y) * log1p(-prediction))
  lambda <- sqrt(2 / variance)
  penalized_beta <- if (length(beta) > 1L) beta[-1L] else beta
  objective <- log_likelihood - lambda * sum(abs(penalized_beta))
  list(
    log_likelihood_check = log_likelihood,
    objective_check = objective,
    prediction = prediction
  )
}

safe_field <- function(obj, name, default = NA) {
  if (is.list(obj) && name %in% names(obj)) {
    return(obj[[name]])
  }
  default
}

fit_counter_fields <- c(
  "sparseLogistic.folds",
  "sparseLogistic.lambdaValues",
  "sparseLogistic.ccdEpochs",
  "sparseLogistic.activeSetOnlyEpochs",
  "sparseLogistic.activeSetFullReconciliationSweeps",
  "sparseLogistic.fullKktScans",
  "sparseLogistic.strongRuleViolations",
  "sparseLogistic.strongRuleNoReference",
  "sparseLogistic.strongRuleReferenceUses",
  "sparseLogistic.strongRuleFullSetBypasses",
  "sparseLogistic.strongRuleScreenedColumns",
  "sparseLogistic.strongRuleScreenedNnz",
  "sparseLogistic.strongRuleKeptCurrentNonzero",
  "sparseLogistic.strongRuleKeptReferenceNonzero",
  "sparseLogistic.strongRuleKeptThresholdNonPositive",
  "sparseLogistic.strongRuleKeptGradient",
  "sparseLogistic.strongRuleActiveKktChecks",
  "sparseLogistic.strongRuleActiveKktFailures",
  "sparseLogistic.strongRuleActiveKktMaxViolation",
  "sparseLogistic.gradientHessianCalls",
  "sparseLogistic.updateCalls",
  "sparseLogistic.kktGradientCalls",
  "sparseLogistic.nnzGradientHessian",
  "sparseLogistic.nnzUpdate",
  "sparseLogistic.nnzKkt",
  "sparseLogistic.gapSafeChecks",
  "sparseLogistic.gapSafeDualFeasibleChecks",
  "sparseLogistic.gapSafeScreenedColumns",
  "sparseLogistic.gapSafeScreenedNnz",
  "sparseLogistic.gapSafeEligibleColumns",
  "sparseLogistic.gapSafeEligibleNnz",
  "sparseLogistic.gapSafeTotalColumns",
  "sparseLogistic.gapSafeTotalNnz",
  "sparseLogistic.foldLocalBuildAttempts",
  "sparseLogistic.foldLocalBuildSuccesses",
  "sparseLogistic.foldLocalBuildFallbacks",
  "sparseLogistic.foldLocalCopyFittedAttempts",
  "sparseLogistic.foldLocalCopyFittedSourceReady",
  "sparseLogistic.foldLocalCopyFittedSuccesses",
  "sparseLogistic.foldLocalInitialComputeXBetaCalls",
  "sparseLogistic.strongSetSizeMean",
  "sparseLogistic.activeSetSizeMean",
  "sparseLogistic.setSizeRecords",
  "sparseLogistic.gradientHessianSeconds",
  "sparseLogistic.updateSeconds",
  "sparseLogistic.kktSeconds",
  "sparseLogistic.screeningSeconds",
  "sparseLogistic.setupSeconds",
  "sparseLogistic.gapSafeSeconds",
  "sparseLogistic.gapSafeMaxScreenedColumns",
  "sparseLogistic.gapSafeMaxScreenedNnz",
  "sparseLogistic.gapSafeLastPrimalObjective",
  "sparseLogistic.gapSafeLastDualObjective",
  "sparseLogistic.gapSafeLastGap",
  "sparseLogistic.gapSafeLastRadius",
  "sparseLogistic.gapSafeLastDualScale",
  "sparseLogistic.gapSafeMaxUnpenalizedGradient",
  "sparseLogistic.foldLocalBuildSeconds",
  "sparseLogistic.foldLocalDataBuildSeconds",
  "sparseLogistic.foldLocalSpecificsBuildSeconds",
  "sparseLogistic.foldLocalCcdSetupSeconds",
  "sparseLogistic.foldLocalInitSeconds",
  "sparseLogistic.foldLocalInitialComputeXBetaSeconds",
  "sparseLogistic.cvBetaSyncSeconds",
  "sparseLogistic.cvScoringSeconds"
)

fit_one <- function(data, kernel, repetition, args) {
  cyclops_data <- data$cyclops_data
  x <- data$x
  y <- data$y
  n_covariates <- Cyclops::getNumberOfCovariates(cyclops_data)
  use_cv <- identical(args$mode, "cv")
  prior <- Cyclops::createPrior(
    "laplace",
    variance = args$variance,
    exclude = 0,
    useCrossValidation = use_cv
  )
  control <- Cyclops::createControl(
    maxIterations = args$max_iterations,
    tolerance = args$tolerance,
    convergenceType = args$convergence_type,
    cvType = args$cv_type,
    fold = if (use_cv) args$folds else 1L,
    lowerLimit = args$lower_limit,
    upperLimit = args$upper_limit,
    gridSteps = args$grid_steps,
    cvRepetitions = 1,
    minCVData = 2,
    noiseLevel = "silent",
    threads = args$threads,
    seed = args$seed,
    resetCoefficients = FALSE,
    logisticKernel = kernel,
    sparseConvergenceType = args$sparse_convergence_type,
    sparseActiveKktTolerance = args$sparse_active_kkt_tolerance,
    sparseActiveKktMovementTolerance = args$sparse_active_kkt_movement_tolerance,
    sparseActiveKktPolishTolerance = args$sparse_active_kkt_polish_tolerance,
    sparseActiveKktCheckEvery = args$sparse_active_kkt_check_every,
    cvPolishPolicy = args$cv_polish_policy,
    cvPolishTolerance = args$cv_polish_tolerance,
    cvPolishMaxIterations = args$cv_polish_max_iterations,
    cvPolishInitialCandidates = args$cv_polish_initial_candidates,
    cvPolishStopProposal = args$cv_polish_stop_proposal
  )

  gc()
  start <- proc.time()
  fit <- Cyclops::fitCyclopsModel(
    cyclopsData = cyclops_data,
    prior = prior,
    control = control,
    forceNewObject = TRUE,
    returnEstimates = TRUE
  )
  elapsed <- proc.time() - start

  beta <- tryCatch(
    stats::coef(fit, ignoreConvergence = TRUE),
    error = function(e) rep(NA_real_, n_covariates)
  )
  variance <- as.numeric(safe_field(fit, "variance", args$variance))[1L]
  metric_x <- x
  metric_beta <- beta
  metric_source <- "fitted"
  if (!is.null(data$reference_x)) {
    metric_x <- data$reference_x
    metric_beta <- expand_duplicate_beta(beta, data, data$reference_data)
    metric_source <- "reference-expanded"
  }
  checked <- if (is.null(metric_x)) {
    compute_prediction_metrics(fit, y, beta, variance)
  } else {
    compute_metrics(metric_x, y, metric_beta, variance)
  }
  penalized_beta <- if (length(beta) == n_covariates) beta[-1L] else beta

  row <- data.frame(
    kernel = kernel,
    repetition = repetition,
    mode = args$mode,
    rows = data$rows,
    features = data$features,
    nnz = data$nnz,
    metric_source = metric_source,
    metric_features = if (is.null(metric_x)) NA_integer_ else ncol(metric_x),
    metric_nnz = if (is.null(metric_x)) NA_real_ else length(metric_x@x),
    elapsed_sec = as.numeric(elapsed["elapsed"]),
    user_sec = as.numeric(elapsed["user.self"]),
    system_sec = as.numeric(elapsed["sys.self"]),
    return_flag = as.character(safe_field(fit, "return_flag", NA_character_)),
    variance = variance,
    log_likelihood = as.numeric(safe_field(fit, "log_likelihood", NA_real_)),
    log_likelihood_check = checked$log_likelihood_check,
    objective_check = checked$objective_check,
    kkt_max_violation = safe_field(checked, "kkt_max_violation", NA_real_),
    kkt_intercept_violation = safe_field(checked, "kkt_intercept_violation", NA_real_),
    kkt_active_violation = safe_field(checked, "kkt_active_violation", NA_real_),
    kkt_inactive_violation = safe_field(checked, "kkt_inactive_violation", NA_real_),
    nonzero = sum(abs(penalized_beta) > 0, na.rm = TRUE),
    beta_l2 = sqrt(sum(beta * beta, na.rm = TRUE)),
    cross_validation = as.character(safe_field(fit, "cross_validation", "")),
    stringsAsFactors = FALSE
  )
  for (field in fit_counter_fields) {
    row[[field]] <- as.numeric(safe_field(fit, field, NA_real_))[1L]
  }
  list(row = row, prediction = checked$prediction)
}

add_comparisons <- function(results, predictions) {
  out <- results
  out$speedup_vs_reference <- NA_real_
  out$objective_diff_vs_reference <- NA_real_
  out$log_likelihood_diff_vs_reference <- NA_real_
  out$prediction_rmse_vs_reference <- NA_real_
  out$prediction_max_abs_diff_vs_reference <- NA_real_
  for (rep in unique(out$repetition)) {
    rows <- which(out$repetition == rep)
    ref <- rows[1L]
    out$speedup_vs_reference[rows] <- out$elapsed_sec[ref] / out$elapsed_sec[rows]
    out$objective_diff_vs_reference[rows] <-
      out$objective_check[rows] - out$objective_check[ref]
    out$log_likelihood_diff_vs_reference[rows] <-
      out$log_likelihood_check[rows] - out$log_likelihood_check[ref]
    ref_prediction <- predictions[[ref]]
    if (length(ref_prediction) == 1L && is.na(ref_prediction)) {
      next
    }
    for (row in rows) {
      prediction <- predictions[[row]]
      if (length(prediction) != length(ref_prediction) || any(is.na(prediction))) {
        next
      }
      diff <- prediction - ref_prediction
      out$prediction_rmse_vs_reference[row] <- sqrt(mean(diff * diff))
      out$prediction_max_abs_diff_vs_reference[row] <- max(abs(diff))
    }
  }
  out
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  if (!args$mode %in% c("fixed", "cv")) {
    stop("--mode must be fixed or cv")
  }
  if (!args$cv_type %in% c("auto", "grid")) {
    stop("--cv-type must be auto or grid")
  }
  if (is.na(args$threads)) {
    args$threads <- args$folds
  }
  if (is.na(args$matrix_rds) && is.na(args$plp_data)) {
    usage(1)
  }
  if (!is.na(args$matrix_rds) && !is.na(args$plp_data)) {
    stop("Use only one of --matrix-rds or --plp-data")
  }

  data <- if (!is.na(args$matrix_rds)) {
    load_rds_input(args$matrix_rds)
  } else {
    load_plp_input(args$plp_data, args$outcome_id, args$sample_size, args$seed)
  }
  if (is.null(data$cyclops_data)) {
    data <- sample_rows(data, args$sample_size, args$seed)
    data$rows <- nrow(data$x)
    data$features <- ncol(data$x)
    data$nnz <- length(data$x@x)
    reference_data <- load_reference_matrix(args$reference_matrix_rds, data$y)
    if (!is.null(reference_data)) {
      data$reference_data <- reference_data
      data$reference_x <- reference_data$x
    }
    cat("Building Cyclops data...\n")
    data$cyclops_data <- make_cyclops_data(data$x, data$y)
  } else {
    reference_data <- load_reference_matrix(args$reference_matrix_rds, data$y)
    if (!is.null(reference_data)) {
      data$reference_data <- reference_data
      data$reference_x <- reference_data$x
    }
  }

  cat("Loaded data: rows=", data$rows,
      " features=", data$features,
      " nnz=", data$nnz,
      " outcomes=", sum(data$y != 0), "\n", sep = "")

  kernels <- trimws(strsplit(args$kernels, ",", fixed = TRUE)[[1L]])
  results <- list()
  for (rep in seq_len(args$repetitions)) {
    for (kernel in kernels) {
      cat("Fitting kernel=", kernel, " repetition=", rep, "...\n", sep = "")
      result <- fit_one(
        data = data,
        kernel = kernel,
        repetition = rep,
        args = args
      )
      results[[length(results) + 1L]] <- result
      print(result$row[, c("kernel", "elapsed_sec", "return_flag",
                           "variance", "nonzero", "kkt_max_violation")])
    }
  }

  out <- add_comparisons(
    do.call(rbind, lapply(results, `[[`, "row")),
    lapply(results, `[[`, "prediction")
  )
  write.csv(out, args$out, row.names = FALSE)
  cat("Wrote ", args$out, "\n", sep = "")
  print(out[, c("kernel", "repetition", "elapsed_sec", "speedup_vs_reference",
                "return_flag", "variance", "nonzero",
                "objective_diff_vs_reference")])
}

tryCatch(main(), error = function(e) {
  message("ERROR: ", conditionMessage(e))
  quit(save = "no", status = 1)
})
