/*
 * AbstractDriver.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: msuchard
 */

#include <numeric>
#include <cmath>
#include <iterator>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <cstdlib>
#include <limits>
#include <string>

// #include "boost/iterator/counting_iterator.hpp"

#include "Types.h"
#include "Thread.h"
#include "AbstractCrossValidationDriver.h"

namespace bsccs {

namespace {

double parsePositiveEnvDouble(const char* name, double defaultValue) {
	const char* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') {
		return defaultValue;
	}
	const double parsed = std::atof(value);
	return parsed > 0.0 ? parsed : defaultValue;
}

int parsePositiveEnvInt(const char* name, int defaultValue) {
	const char* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') {
		return defaultValue;
	}
	const int parsed = std::atoi(value);
	return parsed > 0 ? parsed : defaultValue;
}

bool useCvPolish(const CCDArguments& arguments) {
	const char* policyEnv = std::getenv("CYCLOPS_CV_POLISH_POLICY");
	const std::string policy = policyEnv != nullptr
			? std::string(policyEnv)
			: arguments.crossValidation.polishPolicy;
	const bool legacyEnvPolish = policyEnv == nullptr
			&& std::getenv("CYCLOPS_CV_POLISH_TOLERANCE") != nullptr;
	if (policy == "selective" || (policy == "none" && !legacyEnvPolish)) {
		return false;
	}
	const double configuredTolerance = arguments.crossValidation.polishTolerance > 0.0
			? arguments.crossValidation.polishTolerance
			: arguments.modeFinding.tolerance;
	const double polishTolerance = parsePositiveEnvDouble(
			"CYCLOPS_CV_POLISH_TOLERANCE",
			configuredTolerance);
	return polishTolerance > 0.0 && polishTolerance < arguments.modeFinding.tolerance;
}

uint64_t deltaCounter(uint64_t after, uint64_t before) {
	return after >= before ? after - before : 0;
}

double deltaCounter(double after, double before) {
	return after - before;
}

double meanSetSizeDelta(uint64_t afterTotal, uint64_t beforeTotal,
		uint64_t afterRecords, uint64_t beforeRecords) {
	const uint64_t records = deltaCounter(afterRecords, beforeRecords);
	if (records == 0) {
		return 0.0;
	}
	return static_cast<double>(deltaCounter(afterTotal, beforeTotal)) /
			static_cast<double>(records);
}

void writeFoldTraceHeader(const char* path) {
	std::ofstream output(path);
	output << "step,task,fold,rep,variance,return_flag,pred_log_likelihood,"
		   << "iterations,convergence_value,objective_value,full_kkt_max_violation,"
		   << "ccd_epochs,active_set_only_epochs,active_set_full_reconciliation_sweeps,"
		   << "full_kkt_scans,strong_rule_violations,"
		   << "strong_rule_screened_columns,strong_rule_screened_nnz,"
		   << "strong_rule_reference_uses,strong_rule_no_reference,strong_rule_full_set_bypasses,"
		   << "strong_rule_active_kkt_checks,strong_rule_active_kkt_failures,"
		   << "strong_rule_active_kkt_max_violation,nnz_gradient_hessian,nnz_update,nnz_kkt,"
		   << "gradient_hessian_seconds,update_seconds,kkt_seconds,screening_seconds,setup_seconds,"
		   << "strong_set_size_mean,active_set_size_mean,set_size_records\n";
}

} // namespace

AbstractCrossValidationDriver::AbstractCrossValidationDriver(
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error,
			std::vector<double>* wtsExclude
	) : AbstractDriver(_logger, _error), weightsExclude(wtsExclude) {
	// Do nothing
}

AbstractCrossValidationDriver::~AbstractCrossValidationDriver() {
	// Do nothing
}

void AbstractCrossValidationDriver::resetForOptimal(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& arguments) {

	// turn off syncCV for calculation of single GPU model
	if (ccd.GPU() && arguments.crossValidation.syncCV) {
		ccd.turnOffSyncCV();
	}

	ccd.setWeights(NULL);

    for (int i = 0; i < maxPoint.point.size(); ++i) {
	    ccd.setHyperprior(i, maxPoint.point[i]);
    }
	ccd.resetBeta(); // Cold-start
}

void AbstractCrossValidationDriver::drive(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& allArguments) {

	// TODO Check that selector is type of CrossValidationSelector
//     const auto& arguments = allArguments.crossValidation;

    // Start of new multi-thread set-up
	int nThreads = (allArguments.threads == -1) ?
        bsccs::thread::hardware_concurrency() :
	    allArguments.threads;

    if (nThreads < 1) {
        nThreads = 1;
    }

	std::ostringstream stream2;
	stream2 << "Using " << nThreads << " thread(s)";
	logger->writeLine(stream2);

	if (ccd.GPU() && allArguments.crossValidation.syncCV) {
//	if (false) {
		ccd.turnOnSyncCV(allArguments.crossValidation.foldToCompute);
		nThreads = 1;
	}
/*
	bool streamCV = true;
	if (streamCV) {
		ccd.turnOnStreamCV(allArguments.crossValidation.foldToCompute);
	}
*/
		std::vector<CyclicCoordinateDescent*> ccdPool;
		std::vector<AbstractSelector*> selectorPool;
		foldLocalCcdPool.clear();
		foldLocalValidationCcdPool.clear();
		foldLocalUnavailable.clear();

	ccdPool.push_back(&ccd);
	selectorPool.push_back(&selector);

	for (int i = 1; i < nThreads; ++i) {
		ccdPool.push_back(ccd.clone(allArguments.computeDevice));
		selectorPool.push_back(selector.clone());
	}

	// Check of poor allocation
	bool allocationError = false;
	for (auto element : ccdPool) {
	    if (element == nullptr) {
	        allocationError = true;
	    }
	}

    for (auto element : selectorPool) {
        if (element == nullptr) {
            allocationError = true;
        }
    }

    if (allocationError) {
        std::ostringstream errorStream;
        errorStream << "Memory allocation error in multi-threaded cross validation driver";
        error->throwError(errorStream);
    }
	// End of multi-thread set-up

	// Delegate to auto or grid loop
    maxPoint = doCrossValidationLoop(ccd, selector, allArguments, nThreads, ccdPool, selectorPool);

	for (int i = 1; i < nThreads; ++i) {
		ccd.addSparseLogisticPerformanceCounters(
				ccdPool[i]->getSparseLogisticPerformanceCounters());
	}
		for (auto& foldLocal : foldLocalCcdPool) {
			if (foldLocal) {
				ccd.addSparseLogisticPerformanceCounters(
						foldLocal->getSparseLogisticPerformanceCounters());
			}
		}
		for (auto& foldLocalValidation : foldLocalValidationCcdPool) {
			if (foldLocalValidation) {
				ccd.addSparseLogisticPerformanceCounters(
						foldLocalValidation->getSparseLogisticPerformanceCounters());
			}
		}

	// Clean up
	for (int i = 1; i < nThreads; ++i) {
		delete ccdPool[i];
		delete selectorPool[i];
		}
		foldLocalCcdPool.clear();
		foldLocalValidationCcdPool.clear();
		foldLocalUnavailable.clear();

	// Report results
	std::ostringstream stream1;
	std::ostringstream report;
	stream1 << std::endl;
	stream1 << "Maximum predicted log likelihood (" << maxPoint.value << ") estimated at:" << std::endl;
	report << maxPoint.value;
	stream1 << "\t";
	for (int i = 0; i < maxPoint.point.size(); ++i) {
	    stream1 << maxPoint.point[i] << " ";
	    report << " " << maxPoint.point[i];
	}
	stream1 << "(variance)" << std::endl;
	if (!allArguments.useNormalPrior) {
	    stream1 << "\t";
	    for (int i = 0; i < maxPoint.point.size(); ++i) {
		    double lambda = convertVarianceToHyperparameter(maxPoint.point[i]);
	        stream1 << lambda << " ";
	    }
		stream1 << "(lambda)" << std::endl;
	}
	logger->writeLine(stream1);

	ccd.setCrossValidationInfo(report.str());
}

double AbstractCrossValidationDriver::doCrossValidationStep(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& allArguments,
		int step,
		int nThreads,
		std::vector<CyclicCoordinateDescent*>& ccdPool,
		std::vector<AbstractSelector*>& selectorPool,
		std::vector<double>& predLogLikelihood){

    const auto& arguments = allArguments.crossValidation;
    bool coldStart = allArguments.resetCoefficients;

	predLogLikelihood.resize(arguments.foldToCompute);

	// GPU using syncCV
	//if (false) {
	if (ccd.GPU() && allArguments.crossValidation.syncCV) {
		int repetitions = arguments.foldToCompute / arguments.fold;

		// set weights
		std::vector<std::vector<double>> weightsPool;
		weightsPool.resize(arguments.foldToCompute);
		for (int i = 0; i < repetitions; ++i) {
			selector.permute();
			for (int j = 0; j < arguments.fold; ++j) {
				int index = i * arguments.fold + j;
				selector.getWeights(j, weightsPool[i * arguments.fold + j]);
				if (weightsExclude){
					for(size_t k = 0; k < weightsExclude->size(); k++){
						if (weightsExclude->at(k) == 1.0){
							weightsPool[i * arguments.fold + j][k] = 0.0;
						}
					}
				}
				ccd.setWeights(&weightsPool[i * arguments.fold + j][0], index);
			}
		}

		std::ostringstream stream;
		stream << "Running at " << ccd.getPriorInfo() << " ";
		stream << "Grid-point #" << (step + 1) << " at ";
		std::vector<double> hyperprior = ccd.getHyperprior();
		std::copy(hyperprior.begin(), hyperprior.end(),
				std::ostream_iterator<double>(stream, " "));

		bool write = true;

		if (write) logger->writeLine(stream);

		if (coldStart) {
			ccd.resetBeta();
		}

		ccd.update(allArguments.modeFinding);

		// calculate predictive log likelihoods
		for (int cvIndex = 0; cvIndex < arguments.foldToCompute; cvIndex++) {
			selector.getComplement(weightsPool[cvIndex]);  // TODO THREAD_SAFE
			if (weightsExclude){
				for(int j = 0; j < (int)weightsExclude->size(); j++){
					if(weightsExclude->at(j) == 1.0){
						weightsPool[cvIndex][j] = 0.0;
					}
				}
			}
		}

		std::vector<double> logLikelihood = ccd.getPredictiveLogLikelihood(weightsPool);
		for (int i=0; i<arguments.foldToCompute; i++) {
			predLogLikelihood[i] = logLikelihood[i];
		}

	} else {

		auto& weightsExclude = this->weightsExclude;
		auto& logger = this->logger;
		auto& foldLocalCcdPool = this->foldLocalCcdPool;
		auto& foldLocalValidationCcdPool = this->foldLocalValidationCcdPool;
		auto& foldLocalUnavailable = this->foldLocalUnavailable;
		const char* foldTracePath = std::getenv("CYCLOPS_CV_FOLD_TRACE");
		const bool useFoldTrace = foldTracePath != nullptr && foldTracePath[0] != '\0';
		const char* foldLocalEnv = std::getenv("CYCLOPS_FOLD_LOCAL_CV");
		const bool useFoldLocalCv = foldLocalEnv != nullptr && foldLocalEnv[0] != '\0' &&
				foldLocalEnv[0] != '0' && !ccd.GPU() && weightsExclude == nullptr;
		const char* foldLocalValidationEnv = std::getenv("CYCLOPS_FOLD_LOCAL_VALIDATION");
		const bool useFoldLocalValidation = useFoldLocalCv
				&& foldLocalValidationEnv != nullptr
				&& foldLocalValidationEnv[0] != '\0'
				&& foldLocalValidationEnv[0] != '0';
		std::mutex foldTraceMutex;
		if (useFoldTrace && step == 0) {
			writeFoldTraceHeader(foldTracePath);
		}
		if (useFoldLocalCv && foldLocalCcdPool.size() !=
				static_cast<size_t>(arguments.foldToCompute)) {
			foldLocalCcdPool.clear();
			foldLocalValidationCcdPool.clear();
			foldLocalCcdPool.resize(arguments.foldToCompute);
			foldLocalValidationCcdPool.resize(arguments.foldToCompute);
			foldLocalUnavailable.assign(arguments.foldToCompute, 0);
			if (step == 0) {
				std::ostringstream stream;
				stream << "Using native fold-local CV training and validation data";
				logger->writeLine(stream);
			}
		}

		// auto scheduler = TaskScheduler<decltype(boost::make_counting_iterator(0))>(
		// 		boost::make_counting_iterator(0),
		// 		boost::make_counting_iterator(arguments.foldToCompute),
		// 		nThreads);

		auto scheduler = TaskScheduler<IncrementableIterator<size_t>>(
		    IncrementableIterator<size_t>(0),
		    IncrementableIterator<size_t>(arguments.foldToCompute),
		    nThreads);

		auto oneTask =
				[step, coldStart, nThreads, &ccdPool, &selectorPool,
				 &arguments, &allArguments, &predLogLikelihood,
				 &weightsExclude, &logger //, &lock
				 //    ,&ccd, &selector
					 , &scheduler, foldTracePath, useFoldTrace, useFoldLocalCv,
					 useFoldLocalValidation,
					 &foldLocalCcdPool, &foldLocalValidationCcdPool,
					 &foldLocalUnavailable, &foldTraceMutex
					 ](int task) {

			const auto uniqueId = scheduler.getThreadIndex(task);
			auto ccdTask = ccdPool[uniqueId];
			auto selectorTask = selectorPool[uniqueId];

			// Bring selector up-to-date
			if (task == 0 || nThreads > 1) {
				selectorTask->reseed();
			}
			int i = (nThreads == 1) ? task : 0;
			for ( ; i <= task; ++i) {
				int fold = i % arguments.fold;
				if (fold == 0) {
					selectorTask->permute();
				}
			}

			int fold = task % arguments.fold;
			ccdTask->getFold(fold);

			// Get this fold and update
			std::vector<double> weights; // Task-specific
			selectorTask->getWeights(fold, weights);
			if (weightsExclude){
				for(size_t j = 0; j < weightsExclude->size(); j++){
					if (weightsExclude->at(j) == 1.0){
						weights[j] = 0.0;
					}
				}
			}
			CyclicCoordinateDescent* fitCcdTask = ccdTask;
			CyclicCoordinateDescent* validationCcdTask = ccdTask;
			bool usingFoldLocal = false;
			if (useFoldLocalCv) {
				auto& foldLocal = foldLocalCcdPool[task];
				if (!foldLocal && !foldLocalUnavailable[task]) {
					SparseLogisticPerformanceCounters foldLocalCounter;
					foldLocalCounter.foldLocalBuildAttempts += 1;
					const auto foldLocalBuildStart = std::chrono::steady_clock::now();
					foldLocal.reset(ccdTask->cloneFoldLocal(
							weights, allArguments.computeDevice, &foldLocalCounter));
					const auto foldLocalBuildEnd = std::chrono::steady_clock::now();
					foldLocalCounter.foldLocalBuildSeconds +=
							std::chrono::duration<double>(
									foldLocalBuildEnd - foldLocalBuildStart).count();
					if (!foldLocal) {
						foldLocalCounter.foldLocalBuildFallbacks += 1;
						foldLocalUnavailable[task] = 1;
					} else {
						foldLocalCounter.foldLocalBuildSuccesses += 1;
					}
					ccdTask->addSparseLogisticPerformanceCounters(foldLocalCounter);
				}
				if (foldLocal && !foldLocalUnavailable[task]) {
					fitCcdTask = foldLocal.get();
					fitCcdTask->getFold(fold);
					usingFoldLocal = true;
				}
			}
			if (!usingFoldLocal) {
				ccdTask->setWeights(&weights[0]);
			}

			std::ostringstream stream;
			stream << "Running at " << fitCcdTask->getPriorInfo() << " ";
			stream << "Grid-point #" << (step + 1) << " at ";
			std::vector<double> hyperprior = fitCcdTask->getHyperprior();
			std::copy(hyperprior.begin(), hyperprior.end(),
					std::ostream_iterator<double>(stream, " "));
			stream << "\tFold #" << (fold + 1)
								  << " Rep #" << (task / arguments.fold + 1) << " pred log like = ";

			if (coldStart) {
				fitCcdTask->resetBeta();
			} else if (usingFoldLocal) {
				fitCcdTask->updateWarmStartStateFrom(*ccdTask);
			}

			const auto countersBefore = fitCcdTask->getSparseLogisticPerformanceCounters();
			fitCcdTask->update(allArguments.modeFinding);
			if (fitCcdTask->getUpdateReturnFlag() == SUCCESS && useCvPolish(allArguments)) {
				ModeFindingArguments polishArguments = allArguments.modeFinding;
				polishArguments.sparseConvergenceType = "auto";
				const double configuredTolerance =
						allArguments.crossValidation.polishTolerance > 0.0
						? allArguments.crossValidation.polishTolerance
						: allArguments.modeFinding.tolerance;
				polishArguments.tolerance = parsePositiveEnvDouble(
						"CYCLOPS_CV_POLISH_TOLERANCE",
						configuredTolerance);
				const int configuredMaxIterations =
						allArguments.crossValidation.polishMaxIterations > 0
						? allArguments.crossValidation.polishMaxIterations
						: allArguments.modeFinding.maxIterations;
				polishArguments.maxIterations = parsePositiveEnvInt(
						"CYCLOPS_CV_POLISH_MAX_ITERATIONS",
						configuredMaxIterations);
				fitCcdTask->update(polishArguments);
			}
			const auto countersAfterSolve = fitCcdTask->getSparseLogisticPerformanceCounters();
			SparseLogisticPerformanceCounters foldCounter;
			foldCounter.folds = 1;
			fitCcdTask->addSparseLogisticPerformanceCounters(foldCounter);

			double logLikelihood = std::numeric_limits<double>::quiet_NaN();
			double objectiveValue = std::numeric_limits<double>::quiet_NaN();
			double fullKktMaxViolation = std::numeric_limits<double>::quiet_NaN();
			if (fitCcdTask->getUpdateReturnFlag() == SUCCESS) {
				SparseLogisticPerformanceCounters scoringCounter;

					// Compute predictive loglikelihood for this fold
					if (usingFoldLocal) {
						selectorTask->getComplement(weights);  // TODO THREAD_SAFE
						if (weightsExclude){
							for(int j = 0; j < (int)weightsExclude->size(); j++){
								if(weightsExclude->at(j) == 1.0){
									weights[j] = 0.0;
								}
							}
						}
						CyclicCoordinateDescent* compactValidationCcd = nullptr;
						std::vector<double> compactValidationWeights;
						if (useFoldLocalValidation) {
							auto& foldLocalValidation = foldLocalValidationCcdPool[task];
							if (!foldLocalValidation) {
								SparseLogisticPerformanceCounters validationCounter;
								validationCounter.foldLocalBuildAttempts += 1;
								const auto validationBuildStart = std::chrono::steady_clock::now();
								foldLocalValidation.reset(ccdTask->cloneFoldLocal(
										weights, allArguments.computeDevice, &validationCounter));
								const auto validationBuildEnd = std::chrono::steady_clock::now();
								validationCounter.foldLocalBuildSeconds +=
										std::chrono::duration<double>(
												validationBuildEnd - validationBuildStart).count();
								if (!foldLocalValidation) {
									validationCounter.foldLocalBuildFallbacks += 1;
								} else {
									validationCounter.foldLocalBuildSuccesses += 1;
								}
								ccdTask->addSparseLogisticPerformanceCounters(validationCounter);
							}
							if (foldLocalValidation) {
								compactValidationCcd = foldLocalValidation.get();
								compactValidationCcd->getFold(fold);
							}
						}
						if (compactValidationCcd != nullptr) {
							const auto betaSyncStart = std::chrono::steady_clock::now();
							compactValidationCcd->updateWarmStartStateFrom(*fitCcdTask);
							const auto betaSyncEnd = std::chrono::steady_clock::now();
							scoringCounter.cvBetaSyncSeconds +=
									std::chrono::duration<double>(
											betaSyncEnd - betaSyncStart).count();
							compactValidationWeights.reserve(weights.size());
							for (const auto weight : weights) {
								if (weight != 0.0) {
									compactValidationWeights.push_back(1.0);
								}
							}
							const auto scoringStart = std::chrono::steady_clock::now();
							logLikelihood = compactValidationCcd->getPredictiveLogLikelihoodFromCurrentXBeta(
									compactValidationWeights.data());
							const auto scoringEnd = std::chrono::steady_clock::now();
							scoringCounter.cvScoringSeconds +=
									std::chrono::duration<double>(
											scoringEnd - scoringStart).count();
							const auto betaSyncStart2 = std::chrono::steady_clock::now();
							validationCcdTask->setBeta(fitCcdTask->getBetaVector());
							const auto betaSyncEnd2 = std::chrono::steady_clock::now();
							scoringCounter.cvBetaSyncSeconds +=
									std::chrono::duration<double>(
											betaSyncEnd2 - betaSyncStart2).count();
						} else {
							const auto betaSyncStart = std::chrono::steady_clock::now();
							validationCcdTask->updateBetaFromVector(fitCcdTask->getBetaVector());
							const auto betaSyncEnd = std::chrono::steady_clock::now();
							scoringCounter.cvBetaSyncSeconds +=
									std::chrono::duration<double>(
											betaSyncEnd - betaSyncStart).count();
							const auto scoringStart = std::chrono::steady_clock::now();
							logLikelihood = validationCcdTask->getPredictiveLogLikelihoodFromCurrentXBeta(
									&weights[0]);
							const auto scoringEnd = std::chrono::steady_clock::now();
							scoringCounter.cvScoringSeconds +=
									std::chrono::duration<double>(
											scoringEnd - scoringStart).count();
						}
					} else {
						selectorTask->getComplement(weights);  // TODO THREAD_SAFE
						if (weightsExclude){
							for(int j = 0; j < (int)weightsExclude->size(); j++){
								if(weightsExclude->at(j) == 1.0){
									weights[j] = 0.0;
								}
							}
						}
						const auto scoringStart = std::chrono::steady_clock::now();
						logLikelihood = ccdTask->getNewPredictiveLogLikelihood(&weights[0]);
						const auto scoringEnd = std::chrono::steady_clock::now();
						scoringCounter.cvScoringSeconds +=
								std::chrono::duration<double>(
										scoringEnd - scoringStart).count();
					}
					fitCcdTask->addSparseLogisticPerformanceCounters(scoringCounter);
					objectiveValue = fitCcdTask->getObjectiveFunction(
							allArguments.modeFinding.convergenceType);
				if (useFoldTrace) {
					fullKktMaxViolation = fitCcdTask->computeSparseLogisticFullKktMaxViolation(
							LogisticKernelType::SPARSE_RESIDUAL);
				}

				// Store value
				stream << logLikelihood;
				predLogLikelihood[task] = logLikelihood;
			} else {
				fitCcdTask->resetBeta(); // cold start for stability
				stream << "Not computed";
				predLogLikelihood[task] = std::numeric_limits<double>::quiet_NaN();
			}

			bool write = true;

			if (write) logger->writeLine(stream);

			if (useFoldTrace) {
				const uint64_t setSizeRecords = deltaCounter(
						countersAfterSolve.setSizeRecords, countersBefore.setSizeRecords);
				std::lock_guard<std::mutex> guard(foldTraceMutex);
				std::ofstream output(foldTracePath, std::ios::app);
				output << std::setprecision(17)
					<< step
					<< "," << task
					<< "," << (fold + 1)
					<< "," << (task / arguments.fold + 1)
					<< "," << (hyperprior.empty() ? std::numeric_limits<double>::quiet_NaN() : hyperprior[0])
					<< "," << static_cast<int>(fitCcdTask->getUpdateReturnFlag())
					<< "," << logLikelihood
					<< "," << fitCcdTask->getIterationCount()
					<< "," << fitCcdTask->getLastConvergenceCriterion()
					<< "," << objectiveValue
					<< "," << fullKktMaxViolation
					<< "," << deltaCounter(countersAfterSolve.ccdEpochs, countersBefore.ccdEpochs)
					<< "," << deltaCounter(countersAfterSolve.activeSetOnlyEpochs, countersBefore.activeSetOnlyEpochs)
					<< "," << deltaCounter(countersAfterSolve.activeSetFullReconciliationSweeps,
							countersBefore.activeSetFullReconciliationSweeps)
					<< "," << deltaCounter(countersAfterSolve.fullKktScans, countersBefore.fullKktScans)
					<< "," << deltaCounter(countersAfterSolve.strongRuleViolations, countersBefore.strongRuleViolations)
					<< "," << deltaCounter(countersAfterSolve.strongRuleScreenedColumns, countersBefore.strongRuleScreenedColumns)
					<< "," << deltaCounter(countersAfterSolve.strongRuleScreenedNnz, countersBefore.strongRuleScreenedNnz)
					<< "," << deltaCounter(countersAfterSolve.strongRuleReferenceUses, countersBefore.strongRuleReferenceUses)
					<< "," << deltaCounter(countersAfterSolve.strongRuleNoReference, countersBefore.strongRuleNoReference)
					<< "," << deltaCounter(countersAfterSolve.strongRuleFullSetBypasses, countersBefore.strongRuleFullSetBypasses)
					<< "," << deltaCounter(countersAfterSolve.strongRuleActiveKktChecks, countersBefore.strongRuleActiveKktChecks)
					<< "," << deltaCounter(countersAfterSolve.strongRuleActiveKktFailures, countersBefore.strongRuleActiveKktFailures)
					<< "," << countersAfterSolve.strongRuleActiveKktMaxViolation
					<< "," << deltaCounter(countersAfterSolve.nnzGradientHessian, countersBefore.nnzGradientHessian)
					<< "," << deltaCounter(countersAfterSolve.nnzUpdate, countersBefore.nnzUpdate)
					<< "," << deltaCounter(countersAfterSolve.nnzKkt, countersBefore.nnzKkt)
					<< "," << deltaCounter(countersAfterSolve.gradientHessianSeconds, countersBefore.gradientHessianSeconds)
					<< "," << deltaCounter(countersAfterSolve.updateSeconds, countersBefore.updateSeconds)
					<< "," << deltaCounter(countersAfterSolve.kktSeconds, countersBefore.kktSeconds)
					<< "," << deltaCounter(countersAfterSolve.screeningSeconds, countersBefore.screeningSeconds)
					<< "," << deltaCounter(countersAfterSolve.setupSeconds, countersBefore.setupSeconds)
					<< "," << meanSetSizeDelta(countersAfterSolve.strongSetSizeTotal,
							countersBefore.strongSetSizeTotal,
							countersAfterSolve.setSizeRecords,
							countersBefore.setSizeRecords)
					<< "," << meanSetSizeDelta(countersAfterSolve.activeSetSizeTotal,
							countersBefore.activeSetSizeTotal,
							countersAfterSolve.setSizeRecords,
							countersBefore.setSizeRecords)
					<< "," << setSizeRecords
					<< "\n";
			}
		};

/*
		// Run all task in serial
		for (int i = 0; i < arguments.foldToCompute; i++) {
			oneTask(i);
		}
*/
		// Run all tasks in parallel
		if (nThreads > 1) {
			ccd.getProgressLogger().setConcurrent(true);
		}
		scheduler.execute(oneTask);
		if (nThreads > 1) {
			ccd.getProgressLogger().setConcurrent(false);
			ccd.getProgressLogger().flush();
		}

	}

	double pointEstimate = computePointEstimate(predLogLikelihood);

	return(pointEstimate);
}

double AbstractCrossValidationDriver::computePointEstimate(const std::vector<double>& value) {
	// Mean of log values, ignoring nans
	double total = 0.0;
	int count = 0;
	for (auto x : value) {
		if (x == x) {
			total += x;
			count += 1;
		}
	}
	return total / static_cast<double>(count);
}

double AbstractCrossValidationDriver::computeStDev(const std::vector<double>& value, double mean) {
	// Ignoring nans
	double inner_product = 0.0;
	int count = 0;
	for (auto x : value) {
		if (x == x) {
			inner_product += x * x;
			count += 1;
		}
	}
	return std::sqrt(inner_product / static_cast<double>(count) - mean * mean);
// 	return std::sqrt(std::inner_product(value.begin(), value.end(), value.begin(), 0.0)
// 	/ static_cast<double>(value.size()) - mean * mean);
}


} // namespace
