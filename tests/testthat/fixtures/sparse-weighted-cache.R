args <- commandArgs(TRUE)
.libPaths(c(args[[1]], .libPaths()))
library(Cyclops)
Sys.unsetenv(c("CYCLOPS_FOLD_LOCAL_CV", "CYCLOPS_SPARSE_LOGISTIC_PROFILE"))
set.seed(47)
n <- 1200L
x <- Matrix::Matrix(cbind(intercept = 1, age = runif(n),
                          sex = rbinom(n, 1, 0.5),
                          rare = rbinom(n, 1, 0.05),
                          continuous = rbinom(n, 1, 0.3) * runif(n)), sparse = TRUE)
y <- rbinom(n, 1, plogis(-2 + x[, 2] + x[, 3]))
check <- function(a, b, tolerance = 1e-5) {
    comparison <- all.equal(as.numeric(a), as.numeric(b), tolerance = tolerance)
    if (!isTRUE(comparison)) stop(paste(comparison, collapse = "; "),
                                  "; max absolute difference = ", max(abs(a - b)))
}
run <- function(kernel, cv = FALSE, mixed = FALSE, threads = 1L) {
    if (mixed) {
        data <- createCyclopsData(y = y, modelType = "lr",
            dx = Matrix::Matrix(as.matrix(x[, 1:2]), sparse = FALSE),
            sx = x[, 5, drop = FALSE], ix = x[, 3:4])
    } else {
        data <- createCyclopsData(y = y, sx = x, modelType = "lr")
    }
    ids <- getCovariateIds(data)
    prior <- createPrior("laplace", variance = 0.01, exclude = as.numeric(ids[1]),
                         useCrossValidation = cv)
    control <- createControl(logisticKernel = kernel, tolerance = 2e-6,
                             maxIterations = 3000, cvType = "auto", fold = 10,
                             startingVariance = 0.01, seed = 47, threads = threads,
                             noiseLevel = "silent")
    if (cv) {
        fit <- fitCyclopsModel(data, prior = prior, control = control)
        stopifnot(fit$return_flag == "SUCCESS")
        return(list(beta = coef(fit), variance = fit$variance))
    }
    control$tolerance <- 1e-10
    weights <- list(rep(1, n), as.numeric(seq_len(n) %% 3 != 0),
                    ifelse(seq_len(n) %% 4 == 0, 0, 0.5),
                    as.numeric(seq_len(n) %% 3 != 1), rep(1, n))
    start <- rep(0, ncol(x))
    lapply(weights, function(w) {
        fit <- fitCyclopsModel(data, prior = prior, control = control,
                               weights = w, startingCoefficients = start)
        stopifnot(fit$return_flag == "SUCCESS")
        start <<- as.numeric(coef(fit))
        list(beta = start,
             gradient = Cyclops::gradient(fit),
             hessian = Cyclops:::.cyclopsGetLogLikelihoodHessianDiagonal(
                 data$cyclopsInterfacePtr, ids),
             score = Cyclops:::getCyclopsPredictiveLogLikelihood(fit, 1 - (w > 0)))
    })
}
for (mixed in c(FALSE, TRUE)) {
    legacy <- run("legacy", mixed = mixed)
    sparse <- run("sparse-residual", mixed = mixed)
    for (i in seq_along(legacy)) {
        for (field in names(legacy[[i]])) {
            cat("Checking mixed", mixed, "weight change", i, field, "\n")
            check(sparse[[i]][[field]], legacy[[i]][[field]])
        }
    }
}
for (threads in c(1L, 3L)) {
    legacyCv <- run("legacy", TRUE, threads = threads)
    sparseCv <- run("auto", TRUE, threads = threads)
    check(sparseCv$variance, legacyCv$variance)
    check(sparseCv$beta, legacyCv$beta, 1e-4)
}
cat("PASS weighted cache: repeated weights, warm starts, derivatives, scoring, auto-CV\n")
