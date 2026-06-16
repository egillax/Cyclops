/*
 * AutoSearchCrossValidationDriver.cpp
 *
 *  Created on: Dec 10, 2013
 *      Author: msuchard
 */

#include <iostream>
#include <iomanip>
#include <numeric>
#include <math.h>
#include <cstdlib>
#include <iterator>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>

#include "Types.h"
#include "Thread.h"
#include "AutoSearchCrossValidationDriver.h"
#include "CyclicCoordinateDescent.h"
#include "CrossValidationSelector.h"
#include "AbstractSelector.h"
#include "../utils/HParSearch.h"

// #include "boost/iterator/counting_iterator.hpp"

namespace bsccs {

const static int MAX_STEPS = 50;

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

double parseNonNegativeEnvDouble(const char* name, double defaultValue) {
	const char* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') {
		return defaultValue;
	}
	const double parsed = std::atof(value);
	return parsed >= 0.0 ? parsed : defaultValue;
}

int parseNonNegativeEnvInt(const char* name, int defaultValue) {
	const char* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') {
		return defaultValue;
	}
	const int parsed = std::atoi(value);
	return parsed >= 0 ? parsed : defaultValue;
}

bool useSelectiveCvPolish(const CCDArguments& arguments) {
	const char* policyEnv = std::getenv("CYCLOPS_CV_POLISH_POLICY");
	const std::string policy = policyEnv != nullptr
			? std::string(policyEnv)
			: arguments.crossValidation.polishPolicy;
	if (policy != "selective") {
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

bool useSelectiveStopVerification(const CCDArguments& arguments) {
	const char* value = std::getenv("CYCLOPS_CV_SELECTIVE_POLISH_VERIFY_STOP");
	if (value != nullptr) {
		return value[0] != '\0' && value[0] != '0';
	}
	const char* repairMode = std::getenv("CYCLOPS_CV_SELECTIVE_POLISH_REPAIR_MODE");
	return (repairMode != nullptr && std::string(repairMode) == "score-margin")
			|| (repairMode == nullptr && arguments.crossValidation.polishPolicy == "selective");
}

bool useSelectiveStopRepair() {
	const char* value = std::getenv("CYCLOPS_CV_SELECTIVE_POLISH_REPAIR_STOP");
	return value == nullptr || value[0] == '\0' || value[0] != '0';
}

bool useSelectiveScoreMarginRepair(const CCDArguments& arguments) {
	const char* value = std::getenv("CYCLOPS_CV_SELECTIVE_POLISH_REPAIR_MODE");
	return (value != nullptr && std::string(value) == "score-margin")
			|| (value == nullptr && arguments.crossValidation.polishPolicy == "selective");
}

bool useSelectiveGeometryRepair() {
	const char* value = std::getenv("CYCLOPS_CV_SELECTIVE_POLISH_REPAIR_GEOMETRY");
	if (value != nullptr) {
		return value[0] != '\0' && value[0] != '0';
	}
	return true;
}

bool useSelectiveStopProposalPolish(const CCDArguments& arguments) {
	const char* value = std::getenv("CYCLOPS_CV_SELECTIVE_POLISH_STOP_PROPOSAL");
	if (value != nullptr) {
		return value[0] != '\0' && value[0] != '0';
	}
	return arguments.crossValidation.polishStopProposal;
}

bool containsVariance(const std::vector<double>& variances, double variance) {
	return std::find(variances.begin(), variances.end(), variance) != variances.end();
}

struct CandidateRecord {
	double variance;
	double looseMean;
	double polishedMean;
	bool polished;
};

const CandidateRecord* findBestPolishedCandidate(
		const std::vector<CandidateRecord>& records) {
	const CandidateRecord* best = nullptr;
	for (const auto& record : records) {
		if (!record.polished || record.polishedMean != record.polishedMean) {
			continue;
		}
		if (best == nullptr || record.polishedMean > best->polishedMean) {
			best = &record;
		}
	}
	return best;
}

void recordCandidate(
		std::vector<CandidateRecord>& records,
		double variance,
		double mean,
		bool polished) {
	for (auto& record : records) {
		if (record.variance == variance) {
			if (polished) {
				record.polishedMean = mean;
				record.polished = true;
			} else if (!record.polished) {
				record.looseMean = mean;
			}
			return;
		}
	}
	records.push_back(CandidateRecord{
		variance,
		polished ? std::numeric_limits<double>::quiet_NaN() : mean,
		polished ? mean : std::numeric_limits<double>::quiet_NaN(),
		polished
	});
}

double findUnpolishedVariance(
		const std::vector<CandidateRecord>& records) {
	for (const auto& record : records) {
		if (!record.polished) {
			return record.variance;
		}
	}
	return std::numeric_limits<double>::quiet_NaN();
}

double findScoreSensitiveUnpolishedVariance(
		const std::vector<CandidateRecord>& records,
		double bestSubmittedValue,
		double relativeWindow) {
	const double rankingWindow = std::abs(bestSubmittedValue) * relativeWindow;
	double bestLooseMean = -std::numeric_limits<double>::infinity();
	double bestVariance = std::numeric_limits<double>::quiet_NaN();
	for (const auto& record : records) {
		if (record.polished || record.looseMean != record.looseMean) {
			continue;
		}
		if (record.looseMean >= bestSubmittedValue - rankingWindow
				&& record.looseMean > bestLooseMean) {
			bestLooseMean = record.looseMean;
			bestVariance = record.variance;
		}
	}
	return bestVariance;
}

double candidateMean(const CandidateRecord& record) {
	return record.polished ? record.polishedMean : record.looseMean;
}

double findBracketSensitiveUnpolishedVariance(
		const std::vector<CandidateRecord>& records) {
	if (records.size() < 3) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	std::vector<size_t> order(records.size());
	std::iota(order.begin(), order.end(), 0);
	std::sort(order.begin(), order.end(),
			[&records](size_t lhs, size_t rhs) {
				return records[lhs].variance < records[rhs].variance;
			});

	size_t bestPosition = order.size();
	double bestMean = -std::numeric_limits<double>::infinity();
	for (size_t position = 0; position < order.size(); ++position) {
		const double mean = candidateMean(records[order[position]]);
		if (mean == mean && mean > bestMean) {
			bestMean = mean;
			bestPosition = position;
		}
	}
	if (bestPosition == order.size()
			|| bestPosition == 0
			|| bestPosition + 1 >= order.size()) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	const size_t left = order[bestPosition - 1];
	const size_t right = order[bestPosition + 1];
	if (!records[left].polished) {
		return records[left].variance;
	}
	if (!records[right].polished) {
		return records[right].variance;
	}
	return std::numeric_limits<double>::quiet_NaN();
}

double findSelectiveRepairVariance(
			const std::vector<CandidateRecord>& records,
			double bestSubmittedValue,
			double relativeWindow,
			bool useScoreMarginRepair,
			bool useGeometryRepair) {
		if (useScoreMarginRepair) {
			const double scoreSensitiveVariance = findScoreSensitiveUnpolishedVariance(
					records, bestSubmittedValue, relativeWindow);
			if (scoreSensitiveVariance == scoreSensitiveVariance) {
				return scoreSensitiveVariance;
			}
			if (useGeometryRepair) {
				return findBracketSensitiveUnpolishedVariance(records);
			}
		return std::numeric_limits<double>::quiet_NaN();
	}
	return findUnpolishedVariance(records);
}

CCDArguments makePolishedArguments(const CCDArguments& arguments) {
	CCDArguments polishedArguments = arguments;
	polishedArguments.modeFinding.sparseConvergenceType = "auto";
	const double configuredTolerance = arguments.crossValidation.polishTolerance > 0.0
			? arguments.crossValidation.polishTolerance
			: arguments.modeFinding.tolerance;
	polishedArguments.modeFinding.tolerance = parsePositiveEnvDouble(
			"CYCLOPS_CV_POLISH_TOLERANCE",
			configuredTolerance);
	const int configuredMaxIterations = arguments.crossValidation.polishMaxIterations > 0
			? arguments.crossValidation.polishMaxIterations
			: arguments.modeFinding.maxIterations;
	polishedArguments.modeFinding.maxIterations = parsePositiveEnvInt(
			"CYCLOPS_CV_POLISH_MAX_ITERATIONS",
			configuredMaxIterations);
	return polishedArguments;
}

bool shouldPolishCandidateBeforeLoose(
		int step,
		const CCDArguments& arguments,
		bool defaultInitialBracket) {
	const int minInitial = parseNonNegativeEnvInt(
			"CYCLOPS_CV_SELECTIVE_POLISH_MIN_INITIAL",
			std::max(0, arguments.crossValidation.polishInitialCandidates));
	if (step < minInitial) {
		return true;
	}
	if (step == 2) {
		const char* value = std::getenv("CYCLOPS_CV_SELECTIVE_POLISH_INITIAL_BRACKET");
		const bool polishInitialBracket = value != nullptr
				? value[0] != '\0' && value[0] != '0'
				: defaultInitialBracket;
		if (polishInitialBracket) {
			return true;
		}
	}
	return false;
}

bool shouldPolishCandidate(
		int step,
			const CCDArguments& arguments,
			double pointEstimate,
			bool hasBestSubmittedValue,
			double bestSubmittedValue,
			double relativeWindow,
			bool defaultInitialBracket) {
	if (shouldPolishCandidateBeforeLoose(step, arguments, defaultInitialBracket)) {
		return true;
	}
	if (!hasBestSubmittedValue) {
		return true;
	}
	const double rankingWindow = std::abs(bestSubmittedValue) * relativeWindow;
	return pointEstimate >= bestSubmittedValue - rankingWindow;
}

std::vector<std::string> splitCsvLine(const std::string& line) {
	std::vector<std::string> fields;
	std::stringstream stream(line);
	std::string field;
	while (std::getline(stream, field, ',')) {
		fields.push_back(field);
	}
	return fields;
}

std::vector<double> loadReplayVariances(const std::string& path) {
	std::ifstream input(path.c_str());
	std::vector<double> variances;
	if (!input) {
		return variances;
	}
	std::string line;
	int varianceColumn = 0;
	bool headerChecked = false;
	while (std::getline(input, line)) {
		if (line.empty()) {
			continue;
		}
		const auto fields = splitCsvLine(line);
		if (fields.empty()) {
			continue;
		}
		if (!headerChecked) {
			headerChecked = true;
			for (size_t i = 0; i < fields.size(); ++i) {
				if (fields[i] == "variance") {
					varianceColumn = static_cast<int>(i);
					break;
				}
			}
			char* end = nullptr;
			std::strtod(fields[varianceColumn].c_str(), &end);
			if (end == fields[varianceColumn].c_str()) {
				continue;
			}
		}
		if (varianceColumn >= static_cast<int>(fields.size())) {
			continue;
		}
		char* end = nullptr;
		const double variance = std::strtod(fields[varianceColumn].c_str(), &end);
		if (end != fields[varianceColumn].c_str() && variance > 0.0) {
			variances.push_back(variance);
		}
	}
	return variances;
}

void writeCandidateTraceHeader(std::ofstream& output, int foldToCompute) {
	output << "step,dim,variance,mean,stdev,nextVariance,nextExpected,continue,polished,polish_reason";
	for (int i = 0; i < foldToCompute; ++i) {
		output << ",fold_" << (i + 1);
	}
	output << "\n";
}

void writeCandidateTraceRow(
		std::ofstream& output,
		int step,
		int dim,
		double variance,
		double mean,
		double stdev,
		double nextVariance,
	double nextExpected,
	bool shouldContinue,
	bool polished,
	const std::string& polishReason,
	const std::vector<double>& predLogLikelihood) {
	output << step
		<< "," << dim
		<< "," << variance
		<< "," << mean
		<< "," << stdev
		<< "," << nextVariance
		<< "," << nextExpected
		<< "," << (shouldContinue ? 1 : 0)
		<< "," << (polished ? 1 : 0)
		<< "," << polishReason;
	for (const auto value : predLogLikelihood) {
		output << "," << value;
	}
	output << "\n";
}

} // namespace

AutoSearchCrossValidationDriver::AutoSearchCrossValidationDriver(
			const AbstractModelData& _modelData,
			const CCDArguments& arguments,
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error,
            vector<double>* wtsExclude
			) : AbstractCrossValidationDriver(_logger, _error, wtsExclude), modelData(_modelData),
			maxSteps(MAX_STEPS) {

	// Do anything???
}

AutoSearchCrossValidationDriver::~AutoSearchCrossValidationDriver() {
	// Do nothing
}

void AutoSearchCrossValidationDriver::logResults(const CCDArguments& allArguments) {
    const auto& arguments = allArguments.crossValidation;
	ofstream outLog(arguments.cvFileName.c_str());
	if (!outLog) {
	    std::ostringstream stream;
		stream << "Unable to open log file: " << arguments.cvFileName;
		error->throwError(stream);
	}
	outLog << std::scientific;
	for (int i = 0; i < maxPoint.point.size(); ++i) {
	    outLog << maxPoint.point[i] << " ";
	}
	outLog << std::endl;
	outLog.close();
}

// This is specific to auto-search
MaxPoint AutoSearchCrossValidationDriver::doCrossValidationLoop(
			CyclicCoordinateDescent& ccd,
			AbstractSelector& selector,
			const CCDArguments& allArguments,
			int nThreads,
			std::vector<CyclicCoordinateDescent*>& ccdPool,
			std::vector<AbstractSelector*>& selectorPool) {

    const auto& arguments = allArguments.crossValidation;

	double tryvalue = (arguments.startingVariance > 0) ?
	    arguments.startingVariance :
		modelData.getNormalBasedDefaultVar();

	std::ostringstream stream;
	stream << "Starting var = " << tryvalue;
	if (arguments.startingVariance == -1) {
	    stream << " (default)";
	}
	logger->writeLine(stream);

	const double tolerance = 1E-2; // TODO Make Cyclops argument

	int nDim = ccd.getHyperprior().size();

	std::vector<double> currentOptimal(nDim, tryvalue);
	double currentOptimalValue = -std::numeric_limits<double>::infinity();

	std::ofstream candidateTrace;
	if (const char* tracePath = std::getenv("CYCLOPS_CV_CANDIDATE_TRACE")) {
		if (tracePath[0] != '\0') {
			candidateTrace.open(tracePath);
			if (!candidateTrace) {
				std::ostringstream errorStream;
				errorStream << "Unable to open CYCLOPS_CV_CANDIDATE_TRACE file: " << tracePath;
				error->throwError(errorStream);
			}
			candidateTrace << std::setprecision(17);
			writeCandidateTraceHeader(candidateTrace, arguments.foldToCompute);
		}
	}

	if (const char* replayPath = std::getenv("CYCLOPS_CV_CANDIDATE_REPLAY")) {
		if (replayPath[0] != '\0') {
			const auto replayVariances = loadReplayVariances(replayPath);
			if (replayVariances.empty()) {
				std::ostringstream errorStream;
				errorStream << "No positive variance candidates found in CYCLOPS_CV_CANDIDATE_REPLAY file: "
				            << replayPath;
				error->throwError(errorStream);
			}
			if (nDim != 1) {
				std::ostringstream errorStream;
				errorStream << "CYCLOPS_CV_CANDIDATE_REPLAY currently supports one hyperprior dimension";
				error->throwError(errorStream);
			}

			double bestValue = -std::numeric_limits<double>::infinity();
			double bestVariance = replayVariances.front();
			for (size_t replayStep = 0; replayStep < replayVariances.size(); ++replayStep) {
				const double variance = replayVariances[replayStep];
				ccd.setHyperprior(0, variance);
				selector.reseed();

				std::vector<double> predLogLikelihood;
				const double pointEstimate = doCrossValidationStep(ccd, selector, allArguments,
						static_cast<int>(replayStep), nThreads, ccdPool, selectorPool, predLogLikelihood);
				const double stdDevEstimate = computeStDev(predLogLikelihood, pointEstimate);
					if (candidateTrace) {
						writeCandidateTraceRow(candidateTrace, static_cast<int>(replayStep), 0,
								variance, pointEstimate, stdDevEstimate,
								std::numeric_limits<double>::quiet_NaN(),
								std::numeric_limits<double>::quiet_NaN(),
								replayStep + 1 < replayVariances.size(), false, "replay",
								predLogLikelihood);
					}
				if (pointEstimate > bestValue) {
					bestValue = pointEstimate;
					bestVariance = variance;
				}
			}
			currentOptimal[0] = bestVariance;
			currentOptimalValue = bestValue;
			return MaxPoint{currentOptimal, currentOptimalValue};
		}
	}

	bool globalFinished = false;
	std::vector<double> savedOptimal;

	while (!globalFinished) {

	    if (nDim > 1) {
	        savedOptimal = currentOptimal; // make copy
	    }

	    for (int dim = 0; dim < nDim; ++dim) {

	        // Local search
	        double autoSearchStep = 10.0;
	        if (const char* envStep = std::getenv("CYCLOPS_AUTO_SEARCH_STDSTEP")) {
	            const double parsedStep = std::atof(envStep);
	            if (parsedStep > 1.0) {
	                autoSearchStep = parsedStep;
	            }
	        }
	        const double stopByY = 0.01;
	        UniModalSearch searcher(autoSearchStep, stopByY, log(1.5));

	        int step = 0;
	        bool dimFinished = false;
	        bool hasBestSubmittedValue = false;
	        double bestSubmittedValue = -std::numeric_limits<double>::infinity();
	        bool forceCurrentCandidatePolish = false;
	        bool forceCurrentCandidateIsExistingPoint = false;
	        bool forceCurrentCandidateIsLocalRefinement = false;
	        bool forceCurrentCandidateIsStopProposal = false;
	        bool localRefinementAttempted = false;
	        bool stopProposalAttempted = false;
	        std::vector<double> polishedCandidateVariances;
	        std::vector<CandidateRecord> candidateRecords;

	        while (!dimFinished) {
	            const bool candidateIsLocalRefinement =
			            forceCurrentCandidateIsLocalRefinement;
	            const bool candidateIsStopProposal =
			            forceCurrentCandidateIsStopProposal;
	            const bool candidateIsStopVerification =
			            forceCurrentCandidatePolish && !candidateIsLocalRefinement
			            && !candidateIsStopProposal;
	            const bool candidateIsExistingPointRepair = forceCurrentCandidateIsExistingPoint;

	            ccd.setHyperprior(dim, currentOptimal[dim]);
	            selector.reseed();

	            std::vector<double> predLogLikelihood;
	            const double completedVariance = currentOptimal[dim];
	            bool candidatePolished = false;
	            std::string polishReason = "none";

	            double pointEstimate;
	            double stdDevEstimate;
	            const bool polishBeforeLoose = useSelectiveCvPolish(allArguments) &&
			            !forceCurrentCandidatePolish &&
		            shouldPolishCandidateBeforeLoose(step, allArguments, true);
	            if (useSelectiveCvPolish(allArguments) &&
			            (forceCurrentCandidatePolish || polishBeforeLoose)) {
		            if (forceCurrentCandidatePolish) {
			            polishReason = candidateIsLocalRefinement ? "local-refine" :
				            (candidateIsStopProposal ? "stop-proposed" :
				            (candidateIsExistingPointRepair ? "repair-existing" : "stop-verify"));
		            } else if (step == 0) {
			            polishReason = "initial";
		            } else if (step < 3) {
			            polishReason = "initial-bracket";
		            } else {
			            polishReason = "pre-loose";
		            }
		            const CCDArguments polishedArguments = makePolishedArguments(allArguments);
		            pointEstimate = doCrossValidationStep(ccd, selector, polishedArguments, step,
				            nThreads, ccdPool, selectorPool, predLogLikelihood);
		            stdDevEstimate = computeStDev(predLogLikelihood, pointEstimate);
		            candidatePolished = true;
		            if (!containsVariance(polishedCandidateVariances, completedVariance)) {
			            polishedCandidateVariances.push_back(completedVariance);
		            }
		            forceCurrentCandidatePolish = false;
		            forceCurrentCandidateIsExistingPoint = false;
		            forceCurrentCandidateIsLocalRefinement = false;
		            forceCurrentCandidateIsStopProposal = false;
	            } else {
		            pointEstimate = doCrossValidationStep(ccd, selector, allArguments, step,
				              nThreads, ccdPool, selectorPool, predLogLikelihood);

		            stdDevEstimate = computeStDev(predLogLikelihood, pointEstimate);
			            if (useSelectiveCvPolish(allArguments) &&
					            shouldPolishCandidate(step, allArguments, pointEstimate,
							            hasBestSubmittedValue, bestSubmittedValue, stopByY, true)) {
			            polishReason = "score-window";
			            const CCDArguments polishedArguments = makePolishedArguments(allArguments);
			            ccd.setHyperprior(dim, completedVariance);
			            selector.reseed();
			            std::vector<double> polishedPredLogLikelihood;
			            pointEstimate = doCrossValidationStep(ccd, selector, polishedArguments, step,
					              nThreads, ccdPool, selectorPool, polishedPredLogLikelihood);
			            stdDevEstimate = computeStDev(polishedPredLogLikelihood, pointEstimate);
			            predLogLikelihood.swap(polishedPredLogLikelihood);
			            candidatePolished = true;
			            if (!containsVariance(polishedCandidateVariances, completedVariance)) {
				            polishedCandidateVariances.push_back(completedVariance);
			            }
		            }
	            }
	            recordCandidate(candidateRecords, completedVariance,
			            pointEstimate, candidatePolished);
/*
	            std::cout << "predLikelihoods: ";
	            for (auto x : predLogLikelihood) {
	            	std::cout << x << " ";
	            }
	            std::cout << "\n";
*/
	            std::ostringstream stream;
	            stream << "AvgPred = " << pointEstimate << " with stdev = " << stdDevEstimate << std::endl;
	            if (candidatePolished) {
		            stream << "Polished auto-search candidate before ranking" << std::endl;
	            }
	            searcher.tried(currentOptimal[dim], pointEstimate, stdDevEstimate);
	            if (!hasBestSubmittedValue || pointEstimate > bestSubmittedValue) {
		            bestSubmittedValue = pointEstimate;
		            hasBestSubmittedValue = true;
	            }
	            StepValue next = searcher.step();
	            const bool finitePositiveNextVariance = next.second == next.second &&
			            next.second > 0.0 &&
			            next.second < std::numeric_limits<double>::infinity();
	            const bool nextVarianceAlreadyPolished =
			            containsVariance(polishedCandidateVariances, next.second);
		            const double repairRelativeWindow = parseNonNegativeEnvDouble(
				            "CYCLOPS_CV_SELECTIVE_POLISH_REPAIR_RELATIVE_WINDOW", stopByY);
		            const double unpolishedEvaluatedVariance = findSelectiveRepairVariance(
				            candidateRecords, bestSubmittedValue, repairRelativeWindow,
				            useSelectiveScoreMarginRepair(allArguments),
				            useSelectiveGeometryRepair());
	            const bool hasUnpolishedEvaluatedVariance =
			            unpolishedEvaluatedVariance == unpolishedEvaluatedVariance;
	            const bool forceStopRepair = useSelectiveCvPolish(allArguments) &&
			            useSelectiveStopRepair() &&
			            !next.first && !candidateIsStopVerification &&
			            !candidateIsLocalRefinement &&
			            !candidateIsStopProposal &&
			            hasUnpolishedEvaluatedVariance;
		            const bool forceStopVerification = useSelectiveCvPolish(allArguments) &&
				            useSelectiveStopVerification(allArguments) &&
			            !forceStopRepair &&
			            !candidateIsLocalRefinement &&
			            !candidateIsStopProposal &&
			            !next.first &&
			            (!candidateIsStopVerification || candidateIsExistingPointRepair) &&
			            finitePositiveNextVariance && !nextVarianceAlreadyPolished;
	            const bool scheduleStopProposalPolish =
			            useSelectiveCvPolish(allArguments) &&
			            useSelectiveStopProposalPolish(allArguments) &&
			            candidateIsStopVerification &&
			            !candidateIsExistingPointRepair &&
			            !stopProposalAttempted &&
			            !next.first &&
			            finitePositiveNextVariance &&
			            !nextVarianceAlreadyPolished;
	            double localRefinementVariance =
			            std::numeric_limits<double>::quiet_NaN();
	            if (!scheduleStopProposalPolish &&
			            candidateIsStopVerification && !candidateIsExistingPointRepair &&
			            !next.first && !localRefinementAttempted) {
		            const CandidateRecord* bestPolished =
				            findBestPolishedCandidate(candidateRecords);
		            if (bestPolished != nullptr &&
				            bestPolished->variance != completedVariance) {
			            const double localRefinementWindow = parseNonNegativeEnvDouble(
					            "CYCLOPS_CV_SELECTIVE_POLISH_LOCAL_REFINE_RELATIVE_WINDOW",
					            stopByY);
			            const double minImprovement =
					            std::abs(bestPolished->polishedMean) * localRefinementWindow;
			            const bool materiallyWorseThanBest =
					            bestPolished->polishedMean - pointEstimate > minImprovement;
			            const double reflectedVariance =
					            bestPolished->variance *
					            (bestPolished->variance / completedVariance);
			            if (materiallyWorseThanBest &&
					            reflectedVariance == reflectedVariance &&
					            reflectedVariance > 0.0 &&
					            reflectedVariance < std::numeric_limits<double>::infinity() &&
					            !containsVariance(polishedCandidateVariances, reflectedVariance)) {
				            localRefinementVariance = reflectedVariance;
			            }
		            }
	            }
	            const bool scheduleLocalRefinement =
			            localRefinementVariance == localRefinementVariance;
	            if (candidateTrace) {
		            const bool traceContinues = !candidateIsLocalRefinement &&
				            !candidateIsStopProposal &&
				            (next.first || forceStopRepair || forceStopVerification ||
				            scheduleStopProposalPolish || scheduleLocalRefinement);
		            writeCandidateTraceRow(candidateTrace, step, dim, completedVariance,
	            pointEstimate, stdDevEstimate, next.second, next.expected,
			            traceContinues,
			            candidatePolished, polishReason, predLogLikelihood);
		            }
	            stream << "Completed at " << currentOptimal[dim] << std::endl;
	            stream << "Next point at " << next.second << " with value " << next.expected << " and continue = " << next.first;
	            if (forceStopRepair) {
		            stream << std::endl << "Repairing unpolished auto-search bracket variance " << unpolishedEvaluatedVariance;
	            }
	            if (forceStopVerification) {
		            stream << std::endl << "Verifying proposed stopping variance with polished CV score";
	            }
	            if (scheduleStopProposalPolish) {
		            stream << std::endl << "Checking final proposed stopping variance "
				            << next.second << " with polished CV score";
	            }
	            if (scheduleLocalRefinement) {
		            stream << std::endl << "Checking reflected local variance "
				            << localRefinementVariance << " with polished CV score";
	            }
	            logger->writeLine(stream);

	            if (candidateIsLocalRefinement || candidateIsStopProposal) {
		            const CandidateRecord* bestPolished =
				            findBestPolishedCandidate(candidateRecords);
		            if (bestPolished != nullptr) {
			            currentOptimal[dim] = bestPolished->variance;
			            currentOptimalValue = bestPolished->polishedMean;
		            } else {
			            currentOptimal[dim] = completedVariance;
			            currentOptimalValue = pointEstimate;
		            }
		            dimFinished = true;
	            } else if (candidateIsStopVerification && candidateIsExistingPointRepair) {
		            currentOptimal[dim] = next.second;
		            currentOptimalValue = next.expected;
	            } else if (candidateIsStopVerification && !next.first) {
		            if (scheduleStopProposalPolish) {
			            currentOptimal[dim] = next.second;
			            forceCurrentCandidatePolish = true;
			            forceCurrentCandidateIsExistingPoint = false;
			            forceCurrentCandidateIsLocalRefinement = false;
			            forceCurrentCandidateIsStopProposal = true;
			            stopProposalAttempted = true;
		            } else if (scheduleLocalRefinement) {
			            currentOptimal[dim] = localRefinementVariance;
			            forceCurrentCandidatePolish = true;
			            forceCurrentCandidateIsExistingPoint = false;
			            forceCurrentCandidateIsLocalRefinement = true;
			            forceCurrentCandidateIsStopProposal = false;
			            localRefinementAttempted = true;
		            } else {
			            const CandidateRecord* bestPolished =
					            findBestPolishedCandidate(candidateRecords);
			            if (bestPolished != nullptr) {
				            currentOptimal[dim] = bestPolished->variance;
				            currentOptimalValue = bestPolished->polishedMean;
			            } else {
				            currentOptimal[dim] = completedVariance;
				            currentOptimalValue = pointEstimate;
			            }
			            dimFinished = true;
		            }
	            } else {
		            currentOptimal[dim] = next.second;
		            currentOptimalValue = next.expected;
	            }
	            if (candidateIsLocalRefinement || candidateIsStopProposal ||
			            scheduleStopProposalPolish || scheduleLocalRefinement) {
		            // Selection is complete or the one local refinement is already scheduled.
	            } else if (forceStopRepair) {
		            currentOptimal[dim] = unpolishedEvaluatedVariance;
		            forceCurrentCandidatePolish = true;
		            forceCurrentCandidateIsExistingPoint = true;
		            forceCurrentCandidateIsLocalRefinement = false;
		            forceCurrentCandidateIsStopProposal = false;
	            } else if (forceStopVerification) {
		            forceCurrentCandidatePolish = true;
		            forceCurrentCandidateIsExistingPoint = false;
		            forceCurrentCandidateIsLocalRefinement = false;
		            forceCurrentCandidateIsStopProposal = false;
	            } else if (!next.first) {
	                dimFinished = true;
	            }
	            std::ostringstream stream1;
	            stream1 << searcher;
	            logger->writeLine(stream1);
	            step++;
	            if (step >= maxSteps) {
	                std::ostringstream stream;
	                stream << "Max steps reached!";
	                logger->writeLine(stream);
	                dimFinished = true;
	            }
	        }
	    }

	    if (nDim == 1) {
	        globalFinished = true;
	    } else {

	        double diff = 0.0;
	        for (int i = 0; i < nDim; ++i) {
	            diff += std::abs((currentOptimal[i] - savedOptimal[i]) / savedOptimal[i]);
	        }
	        std::ostringstream stream;
	        stream << "Absolute percent difference in cycle: " << diff << std::endl;

	        globalFinished = (diff < tolerance);
	    }
	}
	return MaxPoint{currentOptimal, currentOptimalValue};
}

} // namespace
