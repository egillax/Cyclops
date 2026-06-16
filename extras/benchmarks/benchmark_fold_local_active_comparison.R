#!/usr/bin/env Rscript

usage <- function(status = 0) {
  cat(
    "Usage:\n",
    "  Rscript extras/benchmarks/benchmark_fold_local_active_comparison.R \\\n",
    "    --matrix-rds data.rds --out fold-local-comparison.csv\n\n",
    "  Rscript extras/benchmarks/benchmark_fold_local_active_comparison.R \\\n",
    "    --plp-data /path/to/plpData --outcome-id 123 --out fold-local-comparison.csv\n\n",
    "Runs two sequential benchmarks and combines results into one CSV:\n",
    "  1. sparse-residual-active\n",
    "  2. sparse-residual-active with CYCLOPS_FOLD_LOCAL_CV=1\n\n",
    "Required input, choose one:\n",
    "  --matrix-rds PATH        RDS list with x=dgCMatrix and y=0/1 vector.\n",
    "  --plp-data PATH          Saved PLP data directory or RDS object.\n\n",
    "Common options:\n",
    "  --out PATH               Combined CSV output path. Default: fold-local-active-comparison.csv\n",
    "  --outcome-id ID          Required for PLP data without embedded population.\n",
    "  --sample-size N          Optional row sample after loading.\n",
    "  --folds N                CV folds. Default: 3\n",
    "  --threads N              Cyclops threads. Default: folds\n",
    "  --seed N                 CV seed. Default: 42\n",
    "  --tolerance VALUE        Cyclops tolerance. Default: 1e-6\n",
    "  --max-iterations N       Max CCD iterations. Default: 1000\n",
    "  --cv-type TYPE           auto or grid. Default: auto\n",
    "  --lower-limit VALUE      CV lower variance bound. Default: 0.01\n",
    "  --upper-limit VALUE      CV upper variance bound. Default: 20\n",
    "  --grid-steps N           Grid steps for grid CV. Default: 10\n",
    "  --benchmark-script PATH  Path to benchmark_plp_logistic_kernels.R.\n",
    "  --work-dir PATH          Directory for intermediate CSVs. Default: OUT_parts\n",
    "  --keep-intermediate TRUE/FALSE  Keep intermediate CSVs. Default: TRUE\n",
    "  --help                   Show this help.\n\n",
    "Advanced: arguments after a standalone -- are passed through to\n",
    "benchmark_plp_logistic_kernels.R.\n",
    sep = ""
  )
  quit(save = "no", status = status)
}

parse_bool <- function(x) {
  tolower(x) %in% c("1", "true", "t", "yes", "y")
}

parse_args <- function(argv) {
  pass_through <- character()
  boundary <- match("--", argv, nomatch = 0L)
  if (boundary > 0L) {
    pass_through <- argv[(boundary + 1L):length(argv)]
    argv <- argv[seq_len(boundary - 1L)]
  }

  args <- list(
    matrix_rds = NA_character_,
    plp_data = NA_character_,
    out = "fold-local-active-comparison.csv",
    outcome_id = NA_integer_,
    sample_size = NA_integer_,
    folds = 3L,
    threads = NA_integer_,
    seed = 42L,
    tolerance = 1e-6,
    max_iterations = 1000L,
    cv_type = "auto",
    lower_limit = 0.01,
    upper_limit = 20,
    grid_steps = 10L,
    benchmark_script = file.path(
      dirname(normalizePath(sys.frame(1)$ofile %||% commandArgs(FALSE)[1], mustWork = FALSE)),
      "benchmark_plp_logistic_kernels.R"
    ),
    work_dir = NA_character_,
    keep_intermediate = TRUE
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
    name <- gsub("-", "_", substring(key, 3L))
    if (!name %in% names(args)) {
      stop("Unknown argument: ", key)
    }
    args[[name]] <- argv[[i + 1L]]
    i <- i + 2L
  }

  int_args <- c("outcome_id", "sample_size", "folds", "threads", "seed",
                "max_iterations", "grid_steps")
  num_args <- c("tolerance", "lower_limit", "upper_limit")
  for (name in int_args) {
    args[[name]] <- if (is.na(args[[name]])) NA_integer_ else as.integer(args[[name]])
  }
  for (name in num_args) {
    args[[name]] <- as.numeric(args[[name]])
  }
  args$keep_intermediate <- if (is.logical(args$keep_intermediate)) {
    args$keep_intermediate
  } else {
    parse_bool(args$keep_intermediate)
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
  if (!args$cv_type %in% c("auto", "grid")) {
    stop("--cv-type must be auto or grid")
  }
  args$pass_through <- pass_through
  args
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L || is.na(x)) y else x
}

script_path <- function() {
  cmd <- commandArgs(FALSE)
  marker <- "--file="
  file_arg <- cmd[startsWith(cmd, marker)][1]
  if (!is.na(file_arg)) {
    return(normalizePath(sub(marker, "", file_arg), mustWork = FALSE))
  }
  normalizePath(sys.frame(1)$ofile %||% "extras/benchmarks/benchmark_fold_local_active_comparison.R",
                mustWork = FALSE)
}

make_child_args <- function(args, out_path) {
  child <- c(
    "--out", out_path,
    "--mode", "cv",
    "--folds", as.character(args$folds),
    "--threads", as.character(args$threads),
    "--seed", as.character(args$seed),
    "--tolerance", as.character(args$tolerance),
    "--max-iterations", as.character(args$max_iterations),
    "--cv-type", args$cv_type,
    "--lower-limit", as.character(args$lower_limit),
    "--upper-limit", as.character(args$upper_limit),
    "--grid-steps", as.character(args$grid_steps),
    "--kernels", "sparse-residual-active",
    "--sparse-convergence-type", "fitted-change",
    "--cv-polish-policy", "selective",
    "--cv-polish-tolerance", "1e-8",
    "--cv-polish-max-iterations", "1500",
    "--cv-polish-initial-candidates", "1",
    "--cv-polish-stop-proposal", "TRUE"
  )
  if (!is.na(args$matrix_rds)) {
    child <- c("--matrix-rds", args$matrix_rds, child)
  } else {
    child <- c("--plp-data", args$plp_data, child)
    if (!is.na(args$outcome_id)) {
      child <- c(child, "--outcome-id", as.character(args$outcome_id))
    }
  }
  if (!is.na(args$sample_size)) {
    child <- c(child, "--sample-size", as.character(args$sample_size))
  }
  c(child, args$pass_through)
}

run_child <- function(label, fold_local, args, out_path) {
  old <- Sys.getenv("CYCLOPS_FOLD_LOCAL_CV", unset = NA_character_)
  on.exit({
    if (is.na(old)) {
      Sys.unsetenv("CYCLOPS_FOLD_LOCAL_CV")
    } else {
      Sys.setenv(CYCLOPS_FOLD_LOCAL_CV = old)
    }
  }, add = TRUE)

  if (fold_local) {
    Sys.setenv(CYCLOPS_FOLD_LOCAL_CV = "1")
  } else {
    Sys.unsetenv("CYCLOPS_FOLD_LOCAL_CV")
  }

  cmd_args <- make_child_args(args, out_path)
  cat("\n== Running ", label, " ==\n", sep = "")
  cat("CYCLOPS_FOLD_LOCAL_CV=", Sys.getenv("CYCLOPS_FOLD_LOCAL_CV", unset = "<unset>"), "\n", sep = "")
  status <- system2(file.path(R.home("bin"), "Rscript"),
                    c(args$benchmark_script, cmd_args))
  if (!identical(status, 0L)) {
    stop("Benchmark failed for ", label, " with status ", status)
  }
  result <- read.csv(out_path, stringsAsFactors = FALSE)
  result$run_label <- label
  result$fold_local <- fold_local
  result$source_csv <- out_path
  result
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  if (is.na(args$work_dir)) {
    stem <- sub("\\.csv$", "", args$out, ignore.case = TRUE)
    args$work_dir <- paste0(stem, "_parts")
  }
  dir.create(dirname(args$out), recursive = TRUE, showWarnings = FALSE)
  dir.create(args$work_dir, recursive = TRUE, showWarnings = FALSE)

  script_default <- file.path(dirname(script_path()), "benchmark_plp_logistic_kernels.R")
  if (!file.exists(args$benchmark_script) && file.exists(script_default)) {
    args$benchmark_script <- script_default
  }
  if (!file.exists(args$benchmark_script)) {
    stop("Cannot find benchmark script: ", args$benchmark_script)
  }

  active_csv <- file.path(args$work_dir, "active-baseline.csv")
  fold_local_csv <- file.path(args$work_dir, "active-fold-local.csv")

  active <- run_child("active-baseline", FALSE, args, active_csv)
  fold_local <- run_child("active-fold-local", TRUE, args, fold_local_csv)
  combined <- rbind(active, fold_local)

  baseline_elapsed <- active$elapsed_sec[[1L]]
  combined$speedup_vs_active_baseline <- baseline_elapsed / combined$elapsed_sec
  combined$elapsed_delta_vs_active_sec <- combined$elapsed_sec - baseline_elapsed

  write.csv(combined, args$out, row.names = FALSE)
  cat("\nWrote combined result: ", args$out, "\n", sep = "")
  print(combined[, intersect(
    c("run_label", "elapsed_sec", "speedup_vs_active_baseline", "return_flag",
      "variance", "nonzero", "kkt_max_violation", "objective_diff_vs_reference"),
    names(combined)
  )])

  if (!args$keep_intermediate) {
    unlink(args$work_dir, recursive = TRUE, force = TRUE)
  }
}

tryCatch(main(), error = function(e) {
  message("ERROR: ", conditionMessage(e))
  quit(save = "no", status = 1)
})
