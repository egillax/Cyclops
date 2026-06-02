#!/usr/bin/env Rscript

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
    "  --kernels LIST           Comma-separated kernels. Default: legacy,sparse-residual\n",
    "  --mode MODE              fixed or cv. Default: cv\n",
    "  --variance VALUE         Fixed-fit Laplace variance. Default: 0.02\n",
    "  --folds N                CV folds. Default: 3\n",
    "  --threads N              Cyclops threads. Default: folds\n",
    "  --seed N                 CV seed. Default: 42\n",
    "  --tolerance VALUE        Cyclops tolerance. Default: 1e-6\n",
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
    kernels = "legacy,sparse-residual",
    mode = "cv",
    variance = 0.02,
    folds = 3L,
    threads = NA_integer_,
    seed = 42L,
    tolerance = 1e-6,
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
                    "repetitions", "sample_size", "outcome_id")
  numeric_args <- c("variance", "tolerance", "lower_limit", "upper_limit")
  for (name in integer_args) {
    args[[name]] <- if (is.na(args[[name]])) NA_integer_ else as.integer(args[[name]])
  }
  for (name in numeric_args) {
    args[[name]] <- as.numeric(args[[name]])
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
    return(list(x = x, y = y, source = "matrix-rds"))
  }
  obj
}

load_plp_input <- function(path, outcome_id) {
  require_package("Matrix")
  require_package("dplyr")
  require_package("PatientLevelPrediction")

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

  row_map <- data.frame(rowId = population$rowId,
                        matrixRowId = seq_len(nrow(population)))
  covariates <- plp_data$covariateData$covariates
  covariates <- dplyr::inner_join(covariates, row_map, by = "rowId", copy = TRUE)
  covariates <- dplyr::select(covariates, matrixRowId, covariateId, covariateValue)
  covariates <- dplyr::collect(covariates)

  feature_ids <- sort(unique(covariates$covariateId))
  col_id <- match(covariates$covariateId, feature_ids)
  x <- Matrix::sparseMatrix(
    i = covariates$matrixRowId,
    j = col_id,
    x = as.numeric(covariates$covariateValue),
    dims = c(length(y), length(feature_ids))
  )
  x <- as(x, "dgCMatrix")

  list(x = x, y = y, source = "plp-data", covariateIds = feature_ids)
}

sample_rows <- function(data, n, seed) {
  if (is.na(n) || n >= length(data$y)) {
    return(data)
  }
  set.seed(seed)
  rows <- sort(sample(seq_along(data$y), n))
  data$x <- data$x[rows, , drop = FALSE]
  data$y <- data$y[rows]
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
  if (is.null(beta) || any(is.na(beta))) {
    return(list(log_likelihood_check = NA_real_, objective_check = NA_real_))
  }

  beta <- as.numeric(beta)
  if (length(beta) == ncol(x) + 1L) {
    eta <- beta[1L] + as.numeric(x %*% beta[-1L])
  } else if (length(beta) == ncol(x)) {
    eta <- as.numeric(x %*% beta)
  } else {
    return(list(log_likelihood_check = NA_real_, objective_check = NA_real_))
  }
  log_likelihood <- sum(y * eta - stable_log1pexp(eta))
  lambda <- sqrt(2 / variance)
  penalized_beta <- if (length(beta) == ncol(x) + 1L) beta[-1L] else beta
  objective <- log_likelihood - lambda * sum(abs(penalized_beta))
  list(
    log_likelihood_check = log_likelihood,
    objective_check = objective
  )
}

safe_field <- function(obj, name, default = NA) {
  if (is.list(obj) && name %in% names(obj)) {
    return(obj[[name]])
  }
  default
}

fit_one <- function(cyclops_data, x, y, kernel, repetition, args) {
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
    convergenceType = "gradient",
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
    logisticKernel = kernel
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
    error = function(e) rep(NA_real_, ncol(x) + 1L)
  )
  variance <- as.numeric(safe_field(fit, "variance", args$variance))[1L]
  checked <- compute_metrics(x, y, beta, variance)
  penalized_beta <- if (length(beta) == ncol(x) + 1L) beta[-1L] else beta

  data.frame(
    kernel = kernel,
    repetition = repetition,
    mode = args$mode,
    rows = nrow(x),
    features = ncol(x),
    nnz = length(x@x),
    elapsed_sec = as.numeric(elapsed["elapsed"]),
    user_sec = as.numeric(elapsed["user.self"]),
    system_sec = as.numeric(elapsed["sys.self"]),
    return_flag = as.character(safe_field(fit, "return_flag", NA_character_)),
    variance = variance,
    log_likelihood = as.numeric(safe_field(fit, "log_likelihood", NA_real_)),
    log_likelihood_check = checked$log_likelihood_check,
    objective_check = checked$objective_check,
    nonzero = sum(abs(penalized_beta) > 0, na.rm = TRUE),
    beta_l2 = sqrt(sum(beta * beta, na.rm = TRUE)),
    cross_validation = as.character(safe_field(fit, "cross_validation", "")),
    stringsAsFactors = FALSE
  )
}

add_comparisons <- function(results) {
  out <- results
  out$speedup_vs_reference <- NA_real_
  out$objective_diff_vs_reference <- NA_real_
  out$log_likelihood_diff_vs_reference <- NA_real_
  for (rep in unique(out$repetition)) {
    rows <- which(out$repetition == rep)
    ref <- rows[1L]
    out$speedup_vs_reference[rows] <- out$elapsed_sec[ref] / out$elapsed_sec[rows]
    out$objective_diff_vs_reference[rows] <-
      out$objective_check[rows] - out$objective_check[ref]
    out$log_likelihood_diff_vs_reference[rows] <-
      out$log_likelihood_check[rows] - out$log_likelihood_check[ref]
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
    load_plp_input(args$plp_data, args$outcome_id)
  }
  data <- sample_rows(data, args$sample_size, args$seed)

  cat("Loaded data: rows=", nrow(data$x),
      " features=", ncol(data$x),
      " nnz=", length(data$x@x),
      " outcomes=", sum(data$y != 0), "\n", sep = "")
  cat("Building Cyclops data...\n")
  cyclops_data <- make_cyclops_data(data$x, data$y)

  kernels <- trimws(strsplit(args$kernels, ",", fixed = TRUE)[[1L]])
  results <- list()
  for (rep in seq_len(args$repetitions)) {
    for (kernel in kernels) {
      cat("Fitting kernel=", kernel, " repetition=", rep, "...\n", sep = "")
      results[[length(results) + 1L]] <- fit_one(
        cyclops_data = cyclops_data,
        x = data$x,
        y = data$y,
        kernel = kernel,
        repetition = rep,
        args = args
      )
      print(results[[length(results)]][, c("kernel", "elapsed_sec", "return_flag",
                                           "variance", "nonzero")])
    }
  }

  out <- add_comparisons(do.call(rbind, results))
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
