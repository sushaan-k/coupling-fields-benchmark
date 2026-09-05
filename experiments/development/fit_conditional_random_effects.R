#!/usr/bin/env Rscript
# Exact hypergeometric-normal ML with the published metafor implementation.
# Usage: Rscript fit_conditional_random_effects.R input.csv output.csv [tolerance]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L || length(args) > 3L) {
  stop("Expected input.csv output.csv [integration tolerance]")
}
library_path <- Sys.getenv("COUPLING_R_LIBRARY", "")
if (nzchar(library_path)) .libPaths(c(library_path, .libPaths()))
if (!requireNamespace("metafor", quietly = TRUE)) stop("metafor is not installed")
tolerance <- if (length(args) == 3L) as.numeric(args[[3]]) else 1e-8
if (!is.finite(tolerance) || tolerance <= 0 || tolerance > 1e-4) {
  stop("Integration tolerance must be in (0, 1e-4]")
}
input <- read.csv(args[[1]], stringsAsFactors = FALSE, check.names = FALSE)
required <- c("donor", "pair", "a", "b", "c", "d")
if (!all(required %in% names(input))) stop("Missing input columns")
counts <- as.matrix(input[c("a", "b", "c", "d")])
if (!is.numeric(counts) || any(!is.finite(counts)) ||
    any(counts < 0) || any(counts != round(counts))) stop("Invalid integer counts")
if (anyDuplicated(input[c("donor", "pair")])) stop("Duplicate donor-pair table")
if (anyNA(input[c("donor", "pair")])) stop("Missing donor or pair identifier")
if (file.exists(args[[2]])) stop("Output already exists")

fit_one <- function(x) {
  warnings <- character()
  fit <- function(relative_tolerance, method = "ML", optimizer = "nlminb", budget = 300L,
                  retry_initializer = FALSE) {
    withCallingHandlers(
      metafor::rma.glmm(
        ai = x$a[informative], bi = x$b[informative],
        ci = x$c[informative], di = x$d[informative],
        measure = "OR", model = "CM.EL", method = method,
        add = 0, to = "none", drop00 = TRUE, skiphet = TRUE,
        control = list(
          optimizer = optimizer, dnchgprec = 1e-12,
          intCtrl = list(rel.tol = relative_tolerance, subdivisions = 500L),
          hessianCtrl = list(r = 4L),
          glmerCtrl = if (retry_initializer) list(nAGQ0initStep = FALSE) else list(),
          optCtrl = if (optimizer == "nlminb")
            list(iter.max = budget, eval.max = 2L * budget, rel.tol = 1e-10) else
            list(maxit = budget, reltol = 1e-10)
        )
      ),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
  }
  row <- data.frame(
    pair = as.character(x$pair[[1]]), mu = NA_real_, tau2 = NA_real_,
    k = NA_integer_, status = "FAILED", warnings = "", error = "",
    package_version = as.character(utils::packageVersion("metafor")),
    hessian_r = 4L, standard_errors_used = FALSE,
    fit_route = "metafor_CM.EL_ML", boundary = 0L, boundary_score_abs = NA_real_,
    quadrature_mu_difference = NA_real_, quadrature_tau2_difference = NA_real_,
    boundary_mu = NA_real_, boundary_loglik = NA_real_, random_loglik = NA_real_,
    optimizer_attempts = "", boundary_variance_score = NA_real_,
    informative_donors = "", uninformative_donors = "",
    variance_profile = "", independent_loglik_difference = NA_real_,
    integration_tolerance = tolerance / 10, stringsAsFactors = FALSE
  )
  tryCatch({
    totals <- x$a + x$b + x$c + x$d
    row_totals <- x$a + x$b
    column_totals <- x$a + x$c
    lower_bounds <- pmax(0, row_totals + column_totals - totals)
    upper_bounds <- pmin(row_totals, column_totals)
    informative <- upper_bounds > lower_bounds
    row$informative_donors <- paste(x$donor[informative], collapse = " | ")
    row$uninformative_donors <- paste(x$donor[!informative], collapse = " | ")
    if (sum(informative) < 2L) stop("Fewer than two informative donor tables")
    at_lower <- all(x$a[informative] == lower_bounds[informative])
    at_upper <- all(x$a[informative] == upper_bounds[informative])
    if (at_lower || at_upper) {
      # The likelihood supremum is one per donor as the mean tends to the
      # common endpoint. The variance is unidentified, not estimated as zero.
      row$boundary <- if (at_upper) 1L else -1L
      row$mu <- row$boundary * Inf
      row$boundary_mu <- row$mu
      row$k <- sum(informative)
      row$boundary_loglik <- 0
      row$random_loglik <- 0
      row$fit_route <- "extended_common_support_endpoint"
      row$status <- "OK"
      return(row)
    }
    boundary <- fit(tolerance / 10, "EE")
    row$boundary_mu <- as.numeric(boundary$beta[[1]])
    row$boundary_loglik <- as.numeric(logLik(boundary))
    row$k <- boundary$k
    identical_tables <- all(vapply(x[c("a", "b", "c", "d")],
                                    function(v) all(v == v[[1]]), logical(1)))
    first <- unlist(x[1, c("a", "b", "c", "d")], use.names = FALSE)
    n <- sum(first)
    r <- first[[1]] + first[[2]]
    c0 <- first[[1]] + first[[3]]
    lower <- max(0, r + c0 - n)
    upper <- min(r, c0)
    if (identical_tables && first[[1]] > lower && first[[1]] < upper) {
      # Every donor likelihood has the same maximum. No mixture can exceed it,
      # so the common conditional MLE with variance zero is the global ML fit.
      support <- seq.int(lower, upper)
      log_mass <- lchoose(c0, support) + lchoose(n - c0, r - support) +
        row$boundary_mu * support
      probability <- exp(log_mass - max(log_mass))
      probability <- probability / sum(probability)
      row$boundary_score_abs <- abs(sum(support * probability) - first[[1]])
      if (row$boundary_score_abs > 1e-7) stop("Identical-table boundary score failed")
      row$mu <- row$boundary_mu
      row$tau2 <- 0
      row$random_loglik <- row$boundary_loglik
      row$fit_route <- "analytic_identical_table_boundary"
      row$status <- "OK"
      row$warnings <- paste(unique(warnings), collapse = " | ")
      return(row)
    }
    records <- lapply(which(informative), function(i) {
      support <- seq.int(lower_bounds[[i]], upper_bounds[[i]])
      list(support = support, observed = x$a[[i]],
           log_weight = lchoose(column_totals[[i]], support) +
             lchoose(totals[[i]] - column_totals[[i]], row_totals[[i]] - support))
    })
    conditional_logp <- function(theta, record) {
      logits <- record$log_weight + theta * record$support
      maximum <- max(logits)
      logits[match(record$observed, record$support)] - maximum - log(sum(exp(logits - maximum)))
    }
    independent_loglik <- function(mu, variance) {
      sum(vapply(records, function(record) {
        if (variance == 0) return(conditional_logp(mu, record))
        integrand <- function(z) vapply(z, function(value) {
          exp(conditional_logp(mu + sqrt(variance) * value, record) + dnorm(value, log = TRUE))
        }, numeric(1))
        integral <- integrate(integrand, -12, 12, rel.tol = 1e-9,
                              abs.tol = 1e-12, subdivisions = 500L)
        log(integral$value)
      }, numeric(1)))
    }
    attempts <- character()
    candidates <- list()
    attempt <- function(optimizer, budget, retry_initializer = FALSE) {
      label <- paste0(optimizer, "[", budget, "]", if (retry_initializer) ":initializer_retry" else "")
      candidate <- tryCatch({
        coarse <- fit(tolerance, optimizer = optimizer, budget = budget,
                      retry_initializer = retry_initializer)
        fine <- fit(tolerance / 10, optimizer = optimizer, budget = budget,
                    retry_initializer = retry_initializer)
        result <- list(mu = as.numeric(fine$beta[[1]]), tau2 = as.numeric(fine$tau2),
                       loglik = as.numeric(logLik(fine)),
                       mu_difference = abs(as.numeric(fine$beta[[1]]) - as.numeric(coarse$beta[[1]])),
                       tau2_difference = abs(fine$tau2 - coarse$tau2),
                       optimizer = optimizer, budget = budget, retry_initializer = retry_initializer)
        attempts <<- c(attempts, sprintf("%s:mu=%.12g,tau2=%.12g,loglik=%.12g",
                                        label, result$mu, result$tau2, result$loglik))
        if (!all(is.finite(c(result$mu, result$tau2, result$loglik))) || result$tau2 < 0)
          stop("Nonfinite or negative estimate")
        if (result$mu_difference > 1e-4 * (1 + abs(result$mu)) ||
            result$tau2_difference > 1e-4 * (1 + result$tau2)) stop("Quadrature instability")
        if (result$loglik < row$boundary_loglik - 1e-6) stop("Below zero-variance boundary likelihood")
        result
      }, error = function(e) {
        attempts <<- c(attempts, sprintf("%s:FAILED:%s", label, conditionMessage(e)))
        NULL
      })
      if (!is.null(candidate)) candidates[[length(candidates) + 1L]] <<- candidate
    }
    attempt("nlminb", 300L)
    if (!length(candidates) || candidates[[1]]$tau2 < 1e-4) {
      attempt("nlminb", 1000L)
      attempt("BFGS", 1000L)
    }
    row$optimizer_attempts <- paste(attempts, collapse = " | ")
    best <- if (length(candidates)) candidates[[which.max(vapply(candidates, `[[`, numeric(1), "loglik"))]] else NULL
    if (is.null(best) || best$tau2 < 1e-4) {
      variance_score <- sum(vapply(records, function(record) {
        logits <- record$log_weight + row$boundary_mu * record$support
        probability <- exp(logits - max(logits))
        probability <- probability / sum(probability)
        mean <- sum(probability * record$support)
        information <- sum(probability * (record$support - mean)^2)
        ((record$observed - mean)^2 - information) / 2
      }, numeric(1)))
      row$boundary_variance_score <- variance_score
      if (is.null(best) && variance_score > -1e-7) {
        attempt("nlminb", 1000L, retry_initializer = TRUE)
        row$optimizer_attempts <- paste(attempts, collapse = " | ")
        if (length(candidates)) best <- candidates[[which.max(vapply(candidates, `[[`, numeric(1), "loglik"))]]
      }
      if (variance_score <= -1e-7) {
        variances <- c(1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16)
        profile <- lapply(variances, function(variance) {
          optimized <- optimize(function(mu) -independent_loglik(mu, variance),
                                row$boundary_mu + c(-12, 12), tol = 1e-7)
          if (abs(optimized$minimum - row$boundary_mu) > 11.99) stop("Variance profile reached mean search boundary")
          c(variance = variance, mu = optimized$minimum, loglik = -optimized$objective)
        })
        row$variance_profile <- paste(vapply(profile, function(v)
          sprintf("%.8g:%.12g:%.12g", v[[1]], v[[2]], v[[3]]), character(1)), collapse = " | ")
        if (max(vapply(profile, `[[`, numeric(1), "loglik")) <= row$boundary_loglik + 1e-6 &&
            (is.null(best) || best$loglik <= row$boundary_loglik + 1e-6)) {
          best <- list(mu = row$boundary_mu, tau2 = 0, loglik = row$boundary_loglik,
                       mu_difference = 0, tau2_difference = 0,
                       optimizer = "profile_checked_boundary", budget = 0)
        }
      }
    }
    if (is.null(best)) stop("No validated exact-ML fit or checked boundary candidate")
    row$independent_loglik_difference <- independent_loglik(best$mu, best$tau2) - best$loglik
    if (abs(row$independent_loglik_difference) > 1e-6) stop("Independent quadrature likelihood disagrees")
    row$mu <- best$mu
    row$tau2 <- best$tau2
    row$random_loglik <- best$loglik
    row$quadrature_mu_difference <- best$mu_difference
    row$quadrature_tau2_difference <- best$tau2_difference
    row$fit_route <- if (best$optimizer == "profile_checked_boundary")
      "profile_checked_zero_variance_boundary" else paste0("metafor_CM.EL_ML_", best$optimizer, "_", best$budget)
    if (isTRUE(best$retry_initializer)) row$fit_route <- paste0(row$fit_route, "_initializer_retry")
    row$status <- "OK"
  }, error = function(e) {
    row$error <<- conditionMessage(e)
  })
  row$warnings <- paste(unique(warnings), collapse = " | ")
  row
}

pairs <- unique(input$pair)
for (index in seq_along(pairs)) {
  result <- fit_one(input[input$pair == pairs[[index]], , drop = FALSE])
  write.table(result, args[[2]], sep = ",", row.names = FALSE,
              col.names = index == 1L, append = index > 1L, na = "")
  message(sprintf("%s: %s (%d/%d)", pairs[[index]], result$status, index, length(pairs)))
}
