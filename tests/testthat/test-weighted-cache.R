test_that("Sparse logistic weighted cache survives auto-CV and weight changes", {
    script <- test_path("fixtures", "sparse-weighted-cache.R")
    rscript <- file.path(R.home("bin"), "Rscript")
    output <- suppressWarnings(system2(rscript,
        c("--vanilla", shQuote(normalizePath(script)),
          shQuote(dirname(find.package("Cyclops")))),
        stdout = TRUE, stderr = TRUE))
    status <- attr(output, "status")
    expect_true(is.null(status) || status == 0L, info = paste(output, collapse = "\n"))
    expect_true(any(grepl("PASS weighted cache", output, fixed = TRUE)))
})
