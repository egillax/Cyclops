library("testthat")

context("test-cv.R")
suppressWarnings(RNGversion("3.5.0"))

test_that("Sparse logistic CV polish controls are validated", {
    control <- createControl(
        logisticKernel = "sparse-residual-active",
        sparseConvergenceType = "fitted-change",
        sparseActiveKktTolerance = 0.15,
        sparseActiveKktMovementTolerance = 1e-9,
        cvPolishPolicy = "selective",
        cvPolishTolerance = 1e-8,
        cvPolishInitialCandidates = 4,
        cvPolishStopProposal = FALSE
    )

    expect_equal(control$sparseConvergenceType, "fitted-change")
    expect_equal(control$sparseActiveKktTolerance, 0.15)
    expect_equal(control$cvPolishPolicy, "selective")
    expect_equal(control$cvPolishTolerance, 1e-8)
    expect_equal(control$cvPolishInitialCandidates, 4)
    expect_false(control$cvPolishStopProposal)
    expect_equal(
        createControl(
            cvPolishPolicy = "selective",
            cvPolishTolerance = 1e-8
        )$cvPolishInitialCandidates,
        1
    )
    expect_equal(
        createControl(logisticKernel = "sparse-residual-active-strong")$logisticKernel,
        "sparse-residual-active-strong"
    )

    expect_error(
        createControl(cvPolishPolicy = "selective"),
        "cvPolishTolerance > 0"
    )
    expect_error(
        createControl(sparseConvergenceType = "unknown"),
        "sparseConvergenceType"
    )
    expect_error(
        createControl(cvPolishInitialCandidates = -1),
        "cvPolishInitialCandidates"
    )
    expect_error(
        createControl(cvPolishInitialCandidates = c(1, 2)),
        "cvPolishInitialCandidates"
    )
    expect_error(
        createControl(cvPolishInitialCandidates = NA_real_),
        "cvPolishInitialCandidates"
    )
    expect_error(
        createControl(cvPolishInitialCandidates = 1.5),
        "cvPolishInitialCandidates"
    )
    expect_error(
        createControl(cvPolishStopProposal = c(TRUE, FALSE)),
        "cvPolishStopProposal"
    )
    expect_error(
        createControl(cvPolishStopProposal = NA),
        "cvPolishStopProposal"
    )
})

test_that("Fold-local sparse logistic CV matches full-data scoring", {
    skip_on_cran()
    suppressWarnings(RNGversion("3.5.0"))
    seed <- 123
    set.seed(seed)

    data <- simulateCyclopsData(
        nstrata = 1,
        nrows = 600,
        ncovars = 20,
        eCovarsPerRow = 2,
        effectSizeSd = 1.5,
        model = "logistic"
    )
    cyclopsData <- convertToCyclopsData(
        data$outcomes,
        data$covariates,
        modelType = "lr",
        addIntercept = TRUE
    )
    prior <- createPrior("laplace", exclude = c(0), useCrossValidation = TRUE)
    control <- createControl(
        noiseLevel = "quiet",
        cvType = "grid",
        fold = 3,
        cvRepetitions = 1,
        seed = seed,
        lowerLimit = 0.01,
        upperLimit = 10,
        gridSteps = 4,
        threads = 3,
        maxIterations = 2000,
        tolerance = 1e-8,
        logisticKernel = "sparse-residual-active"
    )

    oldFoldLocal <- Sys.getenv("CYCLOPS_FOLD_LOCAL_CV", unset = NA_character_)
    on.exit({
        if (is.na(oldFoldLocal)) {
            Sys.unsetenv("CYCLOPS_FOLD_LOCAL_CV")
        } else {
            Sys.setenv(CYCLOPS_FOLD_LOCAL_CV = oldFoldLocal)
        }
    }, add = TRUE)

    Sys.unsetenv("CYCLOPS_FOLD_LOCAL_CV")
    capture.output(
        fullFit <- fitCyclopsModel(
            cyclopsData,
            prior = prior,
            control = control,
            forceNewObject = TRUE
        )
    )

    Sys.setenv(CYCLOPS_FOLD_LOCAL_CV = "1")
    capture.output(
        foldLocalFit <- fitCyclopsModel(
            cyclopsData,
            prior = prior,
            control = control,
            forceNewObject = TRUE
        )
    )

    fullCv <- Cyclops:::getCrossValidationInfo(fullFit)
    foldLocalCv <- Cyclops:::getCrossValidationInfo(foldLocalFit)

    expect_equal(foldLocalFit$variance, fullFit$variance, tolerance = 1e-12)
    expect_equal(foldLocalCv$ordinate, fullCv$ordinate, tolerance = 1e-8)
    expect_equal(foldLocalCv$point, fullCv$point, tolerance = 1e-12)
    expect_equal(
        coef(foldLocalFit, ignoreConvergence = TRUE),
        coef(fullFit, ignoreConvergence = TRUE),
        tolerance = 1e-8
    )
})

# ### COX ERROR
#
# library(Cyclops)
# set.seed(100)
# data <- simulateCyclopsData(nstrata=1,nrows=1000,ncovars=200,model="survival")
# cyclopsData <- convertToCyclopsData(data$outcomes,data$covariates,modelType = "cox")
#
# prior <- createPrior("laplace", useCrossValidation = TRUE)
# control <- createControl(noiseLevel = "quiet",lowerLimit = 0.000001,upperLimit = 100,
#                          cvType = "auto", startingVariance = 0.1,
#                          seed = 100,
#                          selectorType = "default")
# fit <- fitCyclopsModel(cyclopsData,prior=prior,control=control)
#
#
# fit <- fitCyclopsModel(cyclopsData,
#                        prior = createPrior("laplace", variance = 0.0464159),
#                        control=control)
#
# # This generates nan on first eval
# library(Cyclops)
# set.seed(100)
# data <- simulateCyclopsData(nstrata=1,nrows=1000,ncovars=200,model="survival")
# cyclopsData <- convertToCyclopsData(data$outcomes,data$covariates,modelType = "cox")
# prior <- createPrior("laplace", useCrossValidation = TRUE)
# control <- createControl(noiseLevel = "quiet",lowerLimit = 0.0464159,upperLimit = 0.0464159,gridSteps=1,
#                          seed = 100,
#                          selectorType = "default")
# fit <- fitCyclopsModel(cyclopsData,prior=prior,control=control)

test_that("Grid in R and auto-search in C++", {
    skip_on_cran() # Do not run on CRAN
    skip("Do not run")

    seed <- 666
    set.seed(seed)

    nrows <- 1000
    ncovars <- 200

    data <- simulateCyclopsData(nstrata = 1,
                                nrows = nrows,
                                ncovars = ncovars,
                                model = "logistic")

    cyclopsData <- convertToCyclopsData(data$outcomes,
                                        data$covariates,
                                        modelType = "lr",
                                        addIntercept = TRUE)

    prior <- createPrior("laplace", exclude = c(0), useCrossValidation = TRUE)

    control <- createControl(noiseLevel = "quiet",
                              cvType = "auto",
                              cvRepetitions = 1,
                              seed = seed)

    out <- capture.output(fit <- fitCyclopsModel(cyclopsData, prior, control))

    cv <- Cyclops:::getCrossValidationInfo(fit)

    printValue <- as.numeric(gsub(".* log likelihood \\((.*)\\) estimated.*", "\\1",
                                  capture.output(cat(out))))

    expect_equal(cv$ordinate, printValue)

    outerGrid <- c(0.5, 1, 2)
    expect_error(createAutoGridCrossValidationControl(outerGrid, autoPosition = 0), "Auto-position")
    expect_error(createAutoGridCrossValidationControl(outerGrid, autoPosition = 3), "Auto-position")

    expect_error(
        fitCyclopsModel(cyclopsData,
                        prior = prior,
                        control = createAutoGridCrossValidationControl(outerGrid)),
        "Auto-grid"
    )

    prior <-
        createParameterizedPrior(
            c("none", rep("laplace", ncovars)),
            parameterize = function(x) {
                lapply(1:(ncovars + 1),
                       function(i) {
                           if (i < (ncovars / 2)) {
                               return(c(0, x[1]))
                           } else {
                               return(c(0, x[1] * x[2]))
                           }
                       })
            },
            values = c(1, 0.5),
            useCrossValidation = TRUE
        )

    fit <- fitCyclopsModel(cyclopsData,
                    prior = prior,
                    control = createAutoGridCrossValidationControl(outerGrid,
                                                                   autoPosition = 1))
    expect_equal(length(fit$searchResults), 3)
    expect_equal(length(Cyclops:::getCrossValidationInfo(fit)$point), 2)

})

test_that("Specify starting variance with auto-search", {
    skip_on_cran() # Do not run on CRAN
    skip("Do not run")
    seed <- 666
    set.seed(seed)
    ntrain <- 100
    data <- simulateCyclopsData(nstrata = 1,
                         nrows = ntrain,
                         ncovars = 2000,
                         model = "logistic")
    cyclopsData <- convertToCyclopsData(data$outcomes,
                                        data$covariates,
                                        modelType = "lr",
                                        addIntercept = TRUE)
    prior <- createPrior("laplace", exclude = c(0), useCrossValidation = TRUE)
    control1 <- createControl(noiseLevel = "quiet",
                              cvType = "auto",
                              cvRepetitions = 1,
                              seed = seed)
    out1 <- capture.output(fit1 <- fitCyclopsModel(cyclopsData,
                            prior = prior,
                            control = control1))

    control2 <- createControl(noiseLevel = "quiet",
                              cvType = "auto",
                              cvRepetitions = 1,
                              seed = seed,
                              startingVariance = 0.1)

    out2 <- capture.output(fit2 <- fitCyclopsModel(cyclopsData,
                                                   prior = prior,
                                                   control = control2))

    expect_true(grep("default", out1[3]) == 1)
    expect_true(length(grep("default", out2[3])) == 0)
    expect_false(out1[3] == out2[3])
})

test_that("Using multi-core CV", {
    skip_on_cran() # Do not run on CRAN
    skip("Do not run")
    ntest <- 1000
    ntrain <- 1000

    set.seed(666)

    data <- simulateCyclopsData(nstrata=1,nrows=ntest+ntrain,ncovars=2000,model="logistic")
    test <- list(outcomes = data$outcomes[1:ntest,], covariates = data$covariates[data$covariates$rowId %in% data$outcomes$rowId[1:ntest],])
    train <- list(outcomes = data$outcomes[(ntest+1):(ntest+ntrain),], covariates = data$covariates[data$covariates$rowId %in% data$outcomes$rowId[(ntest+1):(ntest+ntrain)],])
    cyclopsData <- convertToCyclopsData(train$outcomes,train$covariates,modelType = "lr",addIntercept = TRUE)
    prior <- createPrior("laplace", exclude = c(0), useCrossValidation = TRUE)

    # 1 thread, cold start
    control <- createControl(noiseLevel = "silent", cvType = "auto", cvRepetitions = 1, seed = 666, threads = 1, resetCoefficients = TRUE)
    time1 <- system.time(fit1 <- fitCyclopsModel(cyclopsData,prior=prior,control=control, forceNewObject = TRUE))

    # 4 thread, cold start
    control <- createControl(noiseLevel = "silent", cvType = "auto", cvRepetitions = 1, seed = 666, threads = 4, resetCoefficients = TRUE)
    time2 <- system.time(fit2 <- fitCyclopsModel(cyclopsData,prior=prior,control=control, forceNewObject = TRUE))

    # 1 thread, warm start
    control <- createControl(noiseLevel = "silent", cvType = "auto", cvRepetitions = 1, seed = 666, threads = 1)
    time3 <- system.time(fit3 <- fitCyclopsModel(cyclopsData,prior=prior,control=control, forceNewObject = TRUE))

    # 4 thread, warm start
    control <- createControl(noiseLevel = "silent", cvType = "auto", cvRepetitions = 1, seed = 666, threads = 4)
    time4 <- system.time(fit4 <- fitCyclopsModel(cyclopsData,prior=prior,control=control,  forceNewObject = TRUE))

    cat("Variance estimates\n")
    cat(fit1$variance,"\n")
    cat(fit2$variance,"\n")
    cat(fit3$variance,"\n")
    cat(fit4$variance,"\n")

    # Cold starting each model fit will return equal single- to multi-core estimates
    expect_equal(fit1$variance, fit2$variance)

    cat("Times\n")
    cat(time1,"\n")
    cat(time2,"\n")
    cat(time3,"\n")
    cat(time4,"\n")

    # Multi-core should be faster
    expect_less_than(time2[3], time1[3])
    # Warm starting should be faster
    expect_less_than(time3[3], time1[3])
})

test_that("Seed gets returned", {
    y <- 0
    x <- 1
    data <- createCyclopsData(y ~ x, modelType = "lr")
    fit <- fitCyclopsModel(data, control = createControl(seed = 123), warnings = FALSE)
    expect_equal(fit$seed, 123)

    fit <- fitCyclopsModel(data, forceNewObject = TRUE,
                           control = createControl(seed = NULL), warnings = FALSE)
    expect_true(!is.null(fit$seed))
})

test_that("Auto cvRepetitions", {
    data <- simulateCyclopsData(nstrata = 1, nrows = 1000, ncovars = 10, model = "logistic")
    cyclopsData <- convertToCyclopsData(data$outcomes,
                                        data$covariates,
                                        modelType = "lr",
                                        addIntercept = TRUE)

    prior <- createPrior("laplace", exclude = c(0), useCrossValidation = TRUE)

    control <- createControl(noiseLevel = "quiet",
                             cvType = "auto",
                             cvRepetitions = "auto")

    fit <- fitCyclopsModel(cyclopsData,
                           prior = prior,
                           control = control)

    expect_equal(fit$cvRepetitions, 4)
})
