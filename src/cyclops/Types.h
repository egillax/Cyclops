/*
 * Types.h
 *
 *  Created on: Aug 20, 2013
 *      Author: msuchard
 */

#ifndef CCD_TYPES_H_
#define CCD_TYPES_H_

#include <cstdint>
#include <vector>
#include <map>
#include <string>
#include <algorithm>

// #if defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L
// C++11
    #include <memory>
    namespace bsccs {
        using std::shared_ptr;
        using std::make_shared;
        using std::unique_ptr;
    }
// #else
// C++98 (R build)
//    #include "boost/smart_ptr.hpp"
//    namespace bsccs {
//        using boost::shared_ptr;
//        using boost::make_shared;
//        using boost::unique_ptr;
//    }
// #endif

#include <unordered_map>
namespace bsccs {
    using std::unordered_map;
}

namespace bsccs {

#if __cplusplus >= 201402L
	using std::make_unique; // Use C++14 version
#else
	// Emulate
	template<typename T, typename ...Args>
	bsccs::unique_ptr<T> make_unique( Args&& ...args ) {
    	return bsccs::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
	}
#endif

// Internal types

// #ifdef DOUBLE_PRECISION
// 	typedef double real;
// #else
// 	typedef float real;
// #endif

// template <typename T>
// using Vector = std::vector<T>;

// template <typename T>
// using VectorPtr = bsccs::shared_ptr<Vector<T> >;

typedef std::vector<int> IntVector;
typedef bsccs::shared_ptr<IntVector> IntVectorPtr;

typedef int64_t IdType;
typedef double Fp64Type;
typedef float Fp32Type;

// Output types

typedef std::pair<std::string,double> ExtraInformation;
typedef std::vector<ExtraInformation> ExtraInformationVector;

struct GapSafeDiagnostic {
	bool available = false;
	bool dualFeasible = false;
	double primalObjective = 0.0;
	double dualObjective = 0.0;
	double dualityGap = 0.0;
	double safeRadius = 0.0;
	double dualScale = 0.0;
	double maxGradientRatio = 0.0;
	double maxUnpenalizedGradient = 0.0;
	uint64_t screenedColumns = 0;
	uint64_t screenedNnz = 0;
	uint64_t eligibleColumns = 0;
	uint64_t eligibleNnz = 0;
	uint64_t totalColumns = 0;
	uint64_t totalNnz = 0;
	double seconds = 0.0;
};

struct SparseLogisticPerformanceCounters {
	uint64_t folds = 0;
	uint64_t lambdaValues = 0;
	uint64_t ccdEpochs = 0;
	uint64_t activeSetOnlyEpochs = 0;
	uint64_t activeSetFullReconciliationSweeps = 0;
	uint64_t fullKktScans = 0;
	uint64_t strongRuleViolations = 0;
	uint64_t strongRuleNoReference = 0;
	uint64_t strongRuleReferenceUses = 0;
	uint64_t strongRuleFullSetBypasses = 0;
	uint64_t strongRuleScreenedColumns = 0;
	uint64_t strongRuleScreenedNnz = 0;
	uint64_t strongRuleKeptCurrentNonzero = 0;
	uint64_t strongRuleKeptReferenceNonzero = 0;
	uint64_t strongRuleKeptThresholdNonPositive = 0;
	uint64_t strongRuleKeptGradient = 0;
	uint64_t strongRuleActiveKktChecks = 0;
	uint64_t strongRuleActiveKktFailures = 0;
	uint64_t gradientHessianCalls = 0;
	uint64_t updateCalls = 0;
	uint64_t kktGradientCalls = 0;
	uint64_t nnzGradientHessian = 0;
	uint64_t nnzUpdate = 0;
	uint64_t nnzKkt = 0;
	uint64_t gapSafeChecks = 0;
	uint64_t gapSafeDualFeasibleChecks = 0;
	uint64_t gapSafeScreenedColumns = 0;
	uint64_t gapSafeScreenedNnz = 0;
	uint64_t gapSafeEligibleColumns = 0;
	uint64_t gapSafeEligibleNnz = 0;
	uint64_t gapSafeTotalColumns = 0;
		uint64_t gapSafeTotalNnz = 0;
		uint64_t foldLocalBuildAttempts = 0;
		uint64_t foldLocalBuildSuccesses = 0;
		uint64_t foldLocalBuildFallbacks = 0;
		uint64_t foldLocalCopyFittedAttempts = 0;
		uint64_t foldLocalCopyFittedSourceReady = 0;
		uint64_t foldLocalCopyFittedSuccesses = 0;
		uint64_t foldLocalInitialComputeXBetaCalls = 0;
		uint64_t strongSetSizeTotal = 0;
	uint64_t activeSetSizeTotal = 0;
	uint64_t setSizeRecords = 0;
	double gradientHessianSeconds = 0.0;
	double updateSeconds = 0.0;
	double kktSeconds = 0.0;
	double screeningSeconds = 0.0;
	double setupSeconds = 0.0;
	double strongRuleActiveKktMaxViolation = 0.0;
	double gapSafeSeconds = 0.0;
	double gapSafeMaxScreenedColumns = 0.0;
	double gapSafeMaxScreenedNnz = 0.0;
	double gapSafeLastPrimalObjective = 0.0;
	double gapSafeLastDualObjective = 0.0;
	double gapSafeLastGap = 0.0;
	double gapSafeLastRadius = 0.0;
	double gapSafeLastDualScale = 0.0;
	double gapSafeMaxUnpenalizedGradient = 0.0;
	double foldLocalBuildSeconds = 0.0;
	double foldLocalDataBuildSeconds = 0.0;
	double foldLocalSpecificsBuildSeconds = 0.0;
	double foldLocalCcdSetupSeconds = 0.0;
	double foldLocalInitSeconds = 0.0;
	double foldLocalInitialComputeXBetaSeconds = 0.0;
	double cvBetaSyncSeconds = 0.0;
	double cvScoringSeconds = 0.0;

	void add(const SparseLogisticPerformanceCounters& other) {
		folds += other.folds;
		lambdaValues += other.lambdaValues;
		ccdEpochs += other.ccdEpochs;
		activeSetOnlyEpochs += other.activeSetOnlyEpochs;
		activeSetFullReconciliationSweeps += other.activeSetFullReconciliationSweeps;
		fullKktScans += other.fullKktScans;
		strongRuleViolations += other.strongRuleViolations;
		strongRuleNoReference += other.strongRuleNoReference;
		strongRuleReferenceUses += other.strongRuleReferenceUses;
		strongRuleFullSetBypasses += other.strongRuleFullSetBypasses;
		strongRuleScreenedColumns += other.strongRuleScreenedColumns;
		strongRuleScreenedNnz += other.strongRuleScreenedNnz;
		strongRuleKeptCurrentNonzero += other.strongRuleKeptCurrentNonzero;
		strongRuleKeptReferenceNonzero += other.strongRuleKeptReferenceNonzero;
		strongRuleKeptThresholdNonPositive += other.strongRuleKeptThresholdNonPositive;
		strongRuleKeptGradient += other.strongRuleKeptGradient;
		strongRuleActiveKktChecks += other.strongRuleActiveKktChecks;
		strongRuleActiveKktFailures += other.strongRuleActiveKktFailures;
		gradientHessianCalls += other.gradientHessianCalls;
		updateCalls += other.updateCalls;
		kktGradientCalls += other.kktGradientCalls;
		nnzGradientHessian += other.nnzGradientHessian;
		nnzUpdate += other.nnzUpdate;
		nnzKkt += other.nnzKkt;
		gapSafeChecks += other.gapSafeChecks;
		gapSafeDualFeasibleChecks += other.gapSafeDualFeasibleChecks;
		gapSafeScreenedColumns += other.gapSafeScreenedColumns;
		gapSafeScreenedNnz += other.gapSafeScreenedNnz;
		gapSafeEligibleColumns += other.gapSafeEligibleColumns;
		gapSafeEligibleNnz += other.gapSafeEligibleNnz;
		gapSafeTotalColumns += other.gapSafeTotalColumns;
		gapSafeTotalNnz += other.gapSafeTotalNnz;
			foldLocalBuildAttempts += other.foldLocalBuildAttempts;
			foldLocalBuildSuccesses += other.foldLocalBuildSuccesses;
			foldLocalBuildFallbacks += other.foldLocalBuildFallbacks;
			foldLocalCopyFittedAttempts += other.foldLocalCopyFittedAttempts;
			foldLocalCopyFittedSourceReady += other.foldLocalCopyFittedSourceReady;
			foldLocalCopyFittedSuccesses += other.foldLocalCopyFittedSuccesses;
			foldLocalInitialComputeXBetaCalls += other.foldLocalInitialComputeXBetaCalls;
			strongSetSizeTotal += other.strongSetSizeTotal;
		activeSetSizeTotal += other.activeSetSizeTotal;
		setSizeRecords += other.setSizeRecords;
		gradientHessianSeconds += other.gradientHessianSeconds;
		updateSeconds += other.updateSeconds;
		kktSeconds += other.kktSeconds;
		screeningSeconds += other.screeningSeconds;
		setupSeconds += other.setupSeconds;
		strongRuleActiveKktMaxViolation = std::max(strongRuleActiveKktMaxViolation,
				other.strongRuleActiveKktMaxViolation);
		gapSafeSeconds += other.gapSafeSeconds;
		gapSafeMaxScreenedColumns = std::max(gapSafeMaxScreenedColumns,
				other.gapSafeMaxScreenedColumns);
		gapSafeMaxScreenedNnz = std::max(gapSafeMaxScreenedNnz,
				other.gapSafeMaxScreenedNnz);
		if (other.gapSafeChecks > 0) {
			gapSafeLastPrimalObjective = other.gapSafeLastPrimalObjective;
			gapSafeLastDualObjective = other.gapSafeLastDualObjective;
			gapSafeLastGap = other.gapSafeLastGap;
			gapSafeLastRadius = other.gapSafeLastRadius;
			gapSafeLastDualScale = other.gapSafeLastDualScale;
		}
		gapSafeMaxUnpenalizedGradient = std::max(gapSafeMaxUnpenalizedGradient,
				other.gapSafeMaxUnpenalizedGradient);
		foldLocalBuildSeconds += other.foldLocalBuildSeconds;
		foldLocalDataBuildSeconds += other.foldLocalDataBuildSeconds;
		foldLocalSpecificsBuildSeconds += other.foldLocalSpecificsBuildSeconds;
		foldLocalCcdSetupSeconds += other.foldLocalCcdSetupSeconds;
		foldLocalInitSeconds += other.foldLocalInitSeconds;
		foldLocalInitialComputeXBetaSeconds += other.foldLocalInitialComputeXBetaSeconds;
		cvBetaSyncSeconds += other.cvBetaSyncSeconds;
		cvScoringSeconds += other.cvScoringSeconds;
	}

	ExtraInformationVector asExtraInformation(const std::string& prefix = "sparseLogistic.") const {
		ExtraInformationVector out;
		out.push_back({prefix + "folds", static_cast<double>(folds)});
		out.push_back({prefix + "lambdaValues", static_cast<double>(lambdaValues)});
		out.push_back({prefix + "ccdEpochs", static_cast<double>(ccdEpochs)});
		out.push_back({prefix + "activeSetOnlyEpochs", static_cast<double>(activeSetOnlyEpochs)});
		out.push_back({prefix + "activeSetFullReconciliationSweeps",
			static_cast<double>(activeSetFullReconciliationSweeps)});
		out.push_back({prefix + "fullKktScans", static_cast<double>(fullKktScans)});
		out.push_back({prefix + "strongRuleViolations", static_cast<double>(strongRuleViolations)});
		out.push_back({prefix + "strongRuleNoReference", static_cast<double>(strongRuleNoReference)});
		out.push_back({prefix + "strongRuleReferenceUses", static_cast<double>(strongRuleReferenceUses)});
		out.push_back({prefix + "strongRuleFullSetBypasses", static_cast<double>(strongRuleFullSetBypasses)});
		out.push_back({prefix + "strongRuleScreenedColumns", static_cast<double>(strongRuleScreenedColumns)});
		out.push_back({prefix + "strongRuleScreenedNnz", static_cast<double>(strongRuleScreenedNnz)});
		out.push_back({prefix + "strongRuleKeptCurrentNonzero", static_cast<double>(strongRuleKeptCurrentNonzero)});
		out.push_back({prefix + "strongRuleKeptReferenceNonzero", static_cast<double>(strongRuleKeptReferenceNonzero)});
		out.push_back({prefix + "strongRuleKeptThresholdNonPositive", static_cast<double>(strongRuleKeptThresholdNonPositive)});
		out.push_back({prefix + "strongRuleKeptGradient", static_cast<double>(strongRuleKeptGradient)});
		out.push_back({prefix + "strongRuleActiveKktChecks", static_cast<double>(strongRuleActiveKktChecks)});
		out.push_back({prefix + "strongRuleActiveKktFailures", static_cast<double>(strongRuleActiveKktFailures)});
		out.push_back({prefix + "strongRuleActiveKktMaxViolation", strongRuleActiveKktMaxViolation});
		out.push_back({prefix + "gradientHessianCalls", static_cast<double>(gradientHessianCalls)});
		out.push_back({prefix + "updateCalls", static_cast<double>(updateCalls)});
		out.push_back({prefix + "kktGradientCalls", static_cast<double>(kktGradientCalls)});
		out.push_back({prefix + "nnzGradientHessian", static_cast<double>(nnzGradientHessian)});
		out.push_back({prefix + "nnzUpdate", static_cast<double>(nnzUpdate)});
		out.push_back({prefix + "nnzKkt", static_cast<double>(nnzKkt)});
		out.push_back({prefix + "gapSafeChecks", static_cast<double>(gapSafeChecks)});
		out.push_back({prefix + "gapSafeDualFeasibleChecks", static_cast<double>(gapSafeDualFeasibleChecks)});
		out.push_back({prefix + "gapSafeScreenedColumns", static_cast<double>(gapSafeScreenedColumns)});
		out.push_back({prefix + "gapSafeScreenedNnz", static_cast<double>(gapSafeScreenedNnz)});
		out.push_back({prefix + "gapSafeEligibleColumns", static_cast<double>(gapSafeEligibleColumns)});
		out.push_back({prefix + "gapSafeEligibleNnz", static_cast<double>(gapSafeEligibleNnz)});
		out.push_back({prefix + "gapSafeTotalColumns", static_cast<double>(gapSafeTotalColumns)});
		out.push_back({prefix + "gapSafeTotalNnz", static_cast<double>(gapSafeTotalNnz)});
			out.push_back({prefix + "foldLocalBuildAttempts", static_cast<double>(foldLocalBuildAttempts)});
			out.push_back({prefix + "foldLocalBuildSuccesses", static_cast<double>(foldLocalBuildSuccesses)});
			out.push_back({prefix + "foldLocalBuildFallbacks", static_cast<double>(foldLocalBuildFallbacks)});
			out.push_back({prefix + "foldLocalCopyFittedAttempts", static_cast<double>(foldLocalCopyFittedAttempts)});
			out.push_back({prefix + "foldLocalCopyFittedSourceReady", static_cast<double>(foldLocalCopyFittedSourceReady)});
			out.push_back({prefix + "foldLocalCopyFittedSuccesses", static_cast<double>(foldLocalCopyFittedSuccesses)});
			out.push_back({prefix + "foldLocalInitialComputeXBetaCalls", static_cast<double>(foldLocalInitialComputeXBetaCalls)});
			out.push_back({prefix + "strongSetSizeMean",
			setSizeRecords == 0 ? 0.0 : static_cast<double>(strongSetSizeTotal) / setSizeRecords});
		out.push_back({prefix + "activeSetSizeMean",
			setSizeRecords == 0 ? 0.0 : static_cast<double>(activeSetSizeTotal) / setSizeRecords});
		out.push_back({prefix + "setSizeRecords", static_cast<double>(setSizeRecords)});
		out.push_back({prefix + "gradientHessianSeconds", gradientHessianSeconds});
		out.push_back({prefix + "updateSeconds", updateSeconds});
		out.push_back({prefix + "kktSeconds", kktSeconds});
		out.push_back({prefix + "screeningSeconds", screeningSeconds});
		out.push_back({prefix + "setupSeconds", setupSeconds});
		out.push_back({prefix + "gapSafeSeconds", gapSafeSeconds});
		out.push_back({prefix + "gapSafeMaxScreenedColumns", gapSafeMaxScreenedColumns});
		out.push_back({prefix + "gapSafeMaxScreenedNnz", gapSafeMaxScreenedNnz});
		out.push_back({prefix + "gapSafeLastPrimalObjective", gapSafeLastPrimalObjective});
		out.push_back({prefix + "gapSafeLastDualObjective", gapSafeLastDualObjective});
		out.push_back({prefix + "gapSafeLastGap", gapSafeLastGap});
		out.push_back({prefix + "gapSafeLastRadius", gapSafeLastRadius});
		out.push_back({prefix + "gapSafeLastDualScale", gapSafeLastDualScale});
		out.push_back({prefix + "gapSafeMaxUnpenalizedGradient", gapSafeMaxUnpenalizedGradient});
		out.push_back({prefix + "foldLocalBuildSeconds", foldLocalBuildSeconds});
		out.push_back({prefix + "foldLocalDataBuildSeconds", foldLocalDataBuildSeconds});
		out.push_back({prefix + "foldLocalSpecificsBuildSeconds", foldLocalSpecificsBuildSeconds});
		out.push_back({prefix + "foldLocalCcdSetupSeconds", foldLocalCcdSetupSeconds});
		out.push_back({prefix + "foldLocalInitSeconds", foldLocalInitSeconds});
		out.push_back({prefix + "foldLocalInitialComputeXBetaSeconds", foldLocalInitialComputeXBetaSeconds});
		out.push_back({prefix + "cvBetaSyncSeconds", cvBetaSyncSeconds});
		out.push_back({prefix + "cvScoringSeconds", cvScoringSeconds});
		return out;
	}
};

struct ProfileInformation {
	bool defined;
	double lower95Bound;
	double upper95Bound;
	int evaluations;

	ProfileInformation() : defined(false), lower95Bound(0.0), upper95Bound(0.0), evaluations(0) { }
	ProfileInformation(double lower, double upper) : defined(true), lower95Bound(lower),
			upper95Bound(upper), evaluations(0) { }
	ProfileInformation(double lower, double upper, int evals) : defined(true), lower95Bound(lower),
			upper95Bound(upper), evaluations(evals) { }
};

typedef std::map<IdType, ProfileInformation> ProfileInformationMap;
typedef std::vector<ProfileInformation> ProfileInformationList;

namespace priors {

enum PriorType {
	NONE = 0,
	LAPLACE,
	NORMAL,
	BAR_UPDATE,
	JEFFREYS
};

} // namespace priors

enum ConvergenceType {
	GRADIENT,
	LANGE,
	MITTAL,
	ONE_STEP,
	ZHANG_OLES
};

enum NoiseLevels {
	SILENT = 0,
	QUIET,
	NOISY
};

enum UpdateReturnFlags {
	SUCCESS = 0,
	FAIL,
	MAX_ITERATIONS,
	ILLCONDITIONED,
	MISSING_COVARIATES,
	POOR_BLR_STEP
};

typedef std::vector<IdType> ProfileVector;

enum class ModelType {
	NONE = 0,
	NORMAL,
	POISSON,
	LOGISTIC,
	CONDITIONAL_LOGISTIC,
	TIED_CONDITIONAL_LOGISTIC,
	EFRON_CONDITIONAL_LOGISTIC,
	CONDITIONAL_POISSON,
	SELF_CONTROLLED_MODEL,
	COX,
	COX_RAW,
	TIME_VARYING_COX,
	FINE_GRAY, // ESK: Added Fine-Gray
	SIZE_OF_ENUM // Keep at end
};

enum class AlgorithmType {
	CCD = 0,
	MM,
	SIZE_OF_ENUM // Keep at end
};

enum class PrecisionType {
	FP64 = 0,
	FP32,
	FP16,
	SIZE_OF_ENUM // Keep at end
};

enum class MmBoundType {
	SHADDOX = 0,
	METHOD_1,
	METHOD_2,
	SIZE_OF_ENUM // Keep at end
};

enum class DeviceType {
    CPU = 0,
    GPU,
    SIZE_OF_ENUM // Keep at end
};

enum class SelectorType {
	DEFAULT,
	BY_PID,
	BY_ROW,
	SIZE_OF_ENUM // Keep at end
};

enum class LogisticKernelType {
	LEGACY,
	SPARSE_RESIDUAL,
	SPARSE_RESIDUAL_ACTIVE,
	SPARSE_RESIDUAL_STRONG,
	SPARSE_RESIDUAL_ACTIVE_STRONG,
	SIZE_OF_ENUM // Keep at end
};

enum class NormalizationType {
    STANDARD_DEVIATION,
    MAX,
    MEDIAN,
    Q95,
    SIZE_OF_ENUM // Keep at end
};

namespace Models {

inline bool removeIntercept(const ModelType modelType) {
	return (modelType == ModelType::CONDITIONAL_LOGISTIC ||
			modelType == ModelType::TIED_CONDITIONAL_LOGISTIC ||
			modelType == ModelType::CONDITIONAL_POISSON ||
			modelType == ModelType::SELF_CONTROLLED_MODEL);
}

inline bool requiresStratumID(const ModelType modelType) {
	return (modelType == ModelType::CONDITIONAL_LOGISTIC ||
			modelType == ModelType::TIED_CONDITIONAL_LOGISTIC ||
            modelType == ModelType::CONDITIONAL_POISSON ||
            modelType == ModelType::SELF_CONTROLLED_MODEL);
}

inline bool requiresCensoredData(const ModelType modelType) {
	return (modelType == ModelType::COX ||
			modelType == ModelType::COX_RAW);
}

inline bool requiresOffset(const ModelType modelType) {
	return (modelType == ModelType::SELF_CONTROLLED_MODEL);
}

} // namespace Models

// Hierarchical prior types
typedef std::vector<int> HierarchicalParentMap;
typedef std::vector<std::vector<int> > HierarchicalChildMap;
typedef std::map<IdType, ProfileVector> NeighborhoodMap;

} // namespace bsccs

#endif /* CCD_TYPES_H_ */
