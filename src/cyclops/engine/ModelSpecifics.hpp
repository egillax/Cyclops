/*
 * ModelSpecifics.hpp
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */

#ifndef MODELSPECIFICS_HPP_
#define MODELSPECIFICS_HPP_

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <limits>
#include <deque>
#include <chrono>
#include <fstream>
#include <mutex>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#endif

#include "ModelSpecifics.h"
#include "Iterators.h"

#include "Recursions.hpp"
#include "ParallelLoops.h"
// #include "Ranges.h"

#ifdef USE_RCPP_PARALLEL
#include <tbb/combinable.h>
#include <tbb/parallel_for.h>
#endif

#ifdef CYCLOPS_DEBUG_TIMING
	#include "Timing.h"
#endif
	namespace bsccs {
	    template <typename RealType>
	    const std::string DenseIterator<RealType>::name = "Den";

	    template <typename RealType>
	    const std::string IndicatorIterator<RealType>::name = "Ind";

	    template <typename RealType>
	    const std::string SparseIterator<RealType>::name = "Spa";

	    template <typename RealType>
	    const std::string InterceptIterator<RealType>::name = "Icp";
	}
//#define OLD_WAY
//#define NEW_WAY1
#define NEW_WAY2

//#define USE_BIGNUM
#define USE_LONG_DOUBLE

namespace bsccs {

#ifdef USE_BIGNUM
	typedef bigNum DDouble;
#else
	#ifdef USE_LONG_DOUBLE
		typedef long double DDouble;
	#else
		typedef double DDouble;
	#endif
#endif

#if defined(DEBUG_COX) || defined(DEBUG_COX_MIN)
    using std::cerr;
    using std::endl;
#endif

inline bool sparseLogisticAvx2GatherEnabled() {
	static const bool enabled = []() {
		const char* env = std::getenv("CYCLOPS_AVX2_GATHER_GH");
		if (env != nullptr && env[0] == '0') {
			return false;
		}
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
		__builtin_cpu_init();
		return __builtin_cpu_supports("avx2") != 0;
#else
		return false;
#endif
	}();
	return enabled;
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2")))
inline void sparseLogisticAvx2IndicatorGradientHessianDouble(
		const int* rows,
		const double* fitted,
		size_t n,
		bool computeHessian,
		double& gradient,
		double& hessian) {
	const __m256d one = _mm256_set1_pd(1.0);
	__m256d gradientVector = _mm256_setzero_pd();
	__m256d hessianVector = _mm256_setzero_pd();
	size_t row = 0;
	for (; row + 3 < n; row += 4) {
		const __m128i rowIndex =
				_mm_loadu_si128(reinterpret_cast<const __m128i*>(rows + row));
		const __m256d p = _mm256_i32gather_pd(fitted, rowIndex, 8);
		gradientVector = _mm256_add_pd(gradientVector, p);
		if (computeHessian) {
			hessianVector = _mm256_add_pd(
					hessianVector,
					_mm256_mul_pd(p, _mm256_sub_pd(one, p)));
		}
	}
	double partial[4];
	_mm256_storeu_pd(partial, gradientVector);
	gradient += partial[0] + partial[1] + partial[2] + partial[3];
	if (computeHessian) {
		_mm256_storeu_pd(partial, hessianVector);
		hessian += partial[0] + partial[1] + partial[2] + partial[3];
	}
	for (; row < n; ++row) {
		const double p = fitted[rows[row]];
		gradient += p;
		if (computeHessian) {
			hessian += p * (1.0 - p);
		}
	}
}
#endif

template <typename RealType>
inline bool sparseLogisticTryAvx2IndicatorGradientHessian(
		const int*,
		const RealType*,
		size_t,
		bool,
		RealType&,
		RealType&) {
	return false;
}

inline bool sparseLogisticTryAvx2IndicatorGradientHessian(
		const int* rows,
		const double* fitted,
		size_t n,
		bool computeHessian,
		double& gradient,
		double& hessian) {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
	if (!sparseLogisticAvx2GatherEnabled()) {
		return false;
	}
	sparseLogisticAvx2IndicatorGradientHessianDouble(
			rows, fitted, n, computeHessian, gradient, hessian);
	return true;
#else
	(void) rows;
	(void) fitted;
	(void) n;
	(void) computeHessian;
	(void) gradient;
	(void) hessian;
	return false;
#endif
}

#define NEW_LOOPS

template <class BaseModel,typename RealType>
ModelSpecifics<BaseModel,RealType>::ModelSpecifics(const ModelData<RealType>& input)
	: AbstractModelSpecifics(input), BaseModel(input.getYVectorRef(), input.getTimeVectorRef()),
   modelData(input),
   hX(modelData.getX()),
   sparseLogisticWeightedColumnViewsReady(false),
   sparseLogisticWeightedRowsHaveUnitWeight(false),
   sparseLogisticStateReady(false),
   sparseLogisticXBetaDirty(false),
   sparseLogisticExpXBetaDirty(false),
   sparseIndicesReady(false),
   sparseLogisticPerformanceCounters()
   // hY(input.getYVectorRef()),
   // hOffs(input.getTimeVectorRef())
 //  hPidOriginal(input.getPidVectorRef()), hPid(const_cast<int*>(hPidOriginal.data())),
 //  boundType(MmBoundType::METHOD_2)
//  	threadPool(4,4,1000)
// threadPool(0,0,10)
	{
    	// Do nothing
	// TODO Memory allocation here
	// std::cerr << "ctor ModelSpecifics \n";

#ifdef CYCLOPS_DEBUG_TIMING
	auto now = bsccs::chrono::system_clock::now();
	auto now_c = bsccs::chrono::system_clock::to_time_t(now);
	std::cout << std::endl << "Start: " << std::ctime(&now_c) << std::endl;
#endif


}

template <class BaseModel, typename RealType>
AbstractModelSpecifics* ModelSpecifics<BaseModel,RealType>::clone(ComputeDeviceArguments computeDevice) const {
	return new ModelSpecifics<BaseModel,RealType>(modelData);
}

template <class BaseModel, typename RealType>
double ModelSpecifics<BaseModel,RealType>::getGradientObjective(bool useCrossValidation) {

#ifdef CYCLOPS_DEBUG_TIMING
    auto start = bsccs::chrono::steady_clock::now();
#endif

		RealType criterion = 0;
		if (sparseLogisticXBetaDirty && std::is_same<BaseModel, LogisticRegression<RealType>>::value) {
			if (useCrossValidation) {
				for (const int i : sparseLogisticCaseRows) {
					if (hKWeight[i] != static_cast<RealType>(0.0)) {
						criterion += sparseLogisticLogOdds(sparseLogisticFitted[i]) * hKWeight[i];
					}
				}
			} else {
				for (const int i : sparseLogisticCaseRows) {
					criterion += sparseLogisticLogOdds(sparseLogisticFitted[i]);
				}
			}
		} else if (useCrossValidation) {
			auto& xBeta = getXBeta();
			for (int i = 0; i < K; i++) {
				criterion += xBeta[i] * hY[i] * hKWeight[i];
			}
		} else {
			auto& xBeta = getXBeta();
			for (int i = 0; i < K; i++) {
				criterion += xBeta[i] * hY[i];
            }
		}
#ifdef CYCLOPS_DEBUG_TIMING
    auto end = bsccs::chrono::steady_clock::now();
    ///////////////////////////"
    duration["getGradObj       "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
		return static_cast<double> (criterion);
	}

template <class BaseModel, typename RealType>
void ModelSpecifics<BaseModel,RealType>::printTiming() {

#ifdef CYCLOPS_DEBUG_TIMING

	std::cout << std::endl;
	for (auto& d : duration) {
		std::cout << d.first << " " << d.second << std::endl;
	}
	std::cout << "NEW LOOPS" << std::endl;

#endif
}

template <class BaseModel,typename RealType>
ModelSpecifics<BaseModel,RealType>::~ModelSpecifics() {
	// TODO Memory release here
	// std::cerr << "dtor ModelSpecifics \n";

#ifdef CYCLOPS_DEBUG_TIMING

    printTiming();

	auto now = bsccs::chrono::system_clock::now();
	auto now_c = bsccs::chrono::system_clock::to_time_t(now);
	std::cout << std::endl << "End:   " << std::ctime(&now_c) << std::endl;
#endif

}

template <class BaseModel, typename RealType> template <class IteratorType>
void ModelSpecifics<BaseModel,RealType>::incrementNormsImpl(int index) {

    // TODO should use row-access
	IteratorType it(hX, index);
	for (; it; ++it) {
		const int k = it.index();
		const RealType x = it.value();

		norm[k] += std::abs(x);
	}
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::initializeMmXt() {

#ifdef CYCLOPS_DEBUG_TIMING
    auto start = bsccs::chrono::steady_clock::now();
#endif

    hXt = hX.transpose();

#ifdef CYCLOPS_DEBUG_TIMING
auto end = bsccs::chrono::steady_clock::now();
///////////////////////////"
duration["transpose        "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::initializeMM(
        MmBoundType boundType,
        const std::vector<bool>& fixBeta
    ) {

#ifdef CYCLOPS_DEBUG_TIMING
    auto start = bsccs::chrono::steady_clock::now();
#endif

    // std::cout << "Computing norms..." << std::endl;
    norm.resize(K);
    zeroVector(&norm[0], K);

    for (int j = 0; j < J; ++j) {
    	if (
    	true
    	// !fixBeta[j]
    	) {
			switch(hX.getFormatType(j)) {
				case INDICATOR :
					incrementNormsImpl<IndicatorIterator<RealType>>(j);
					break;
				case SPARSE :
					incrementNormsImpl<SparseIterator<RealType>>(j);
					break;
				case DENSE :
					incrementNormsImpl<DenseIterator<RealType>>(j);
					break;
				case INTERCEPT :
					incrementNormsImpl<InterceptIterator<RealType>>(j);
					break;
			}
        }
    }

    if (boundType == MmBoundType::METHOD_1) {
        // std::cerr << "boundType: METHOD_1" << std::endl;


    } else if (boundType == MmBoundType::METHOD_2) {
        // std::cerr << "boundType: METHOD_2" << std::endl;

        double total = 0;

        for (int j = 0; j < J; ++j) {
            if (!fixBeta[j]) {
                typedef std::pair<double,double> Range;

                std::vector<Range> range(modelData.getNumberOfPatients(), Range(
                        -std::numeric_limits<double>::max(),
                        std::numeric_limits<double>::max()
                ));

                modelData.binaryReductionByStratum(range, j, [](Range& r, double x) {
                    return Range(
                        std::max(x, r.first),
                        std::min(x, r.second)
                    );
                });

                double curvature = 0.0;
                for (Range r : range) {
                    curvature += r.first - r.second;
                }
                curvature /= 4;

                total += curvature;
            }
        }

        if (curvature.size() == 0) {
            curvature.resize(J);
        }

        for (int j = 0; j < J; ++j) {
            curvature[j] = total;
        }
    }

#ifdef CYCLOPS_DEBUG_TIMING
    auto end = bsccs::chrono::steady_clock::now();
    ///////////////////////////"
    duration["norms            "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

// 	struct timeval time1, time2;
// 	gettimeofday(&time1, NULL);
//
//     std::cout << "Constructing Xt..." << std::endl;

//     if (!hXt) {
//         initializeMmXt();
//     }

//     gettimeofday(&time2, NULL);
// 	double duration = //calculateSeconds(time1, time2);
// 		time2.tv_sec - time1.tv_sec +
// 			(double)(time2.tv_usec - time1.tv_usec) / 1000000.0;
//
// 	std::cout << "Done with MM initialization" << std::endl;
// 	std::cout << duration << std::endl;
	// Rcpp::stop("out");

}

template <class BaseModel,typename RealType>
const std::vector<double> ModelSpecifics<BaseModel,RealType>::getXBeta() {
	syncSparseLogisticState();
    return std::vector<double>(std::begin(hXBeta), std::end(hXBeta));
}

template <class BaseModel,typename RealType>
const std::vector<double> ModelSpecifics<BaseModel,RealType>::getXBetaSave() {
    return std::vector<double>(std::begin(hXBetaSave), std::end(hXBetaSave));
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::zeroXBeta() {
	sparseLogisticStateReady = false;
	sparseLogisticXBetaDirty = false;
	sparseLogisticExpXBetaDirty = false;
	std::fill(std::begin(hXBeta), std::end(hXBeta), 0.0);
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::saveXBeta() {
	auto& xBeta = getXBeta();
	if (hXBetaSave.size() < xBeta.size()) {
		hXBetaSave.resize(xBeta.size());
	}
	std::copy(std::begin(xBeta), std::end(xBeta), std::begin(hXBetaSave));
}



template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeXBeta(double* beta, bool useWeights) {
	sparseLogisticStateReady = false;
	sparseLogisticXBetaDirty = false;
	sparseLogisticExpXBetaDirty = false;

    if (!hXt) {
        initializeMmXt();
    }

#ifdef CYCLOPS_DEBUG_TIMING
    auto start = bsccs::chrono::steady_clock::now();
#endif

    switch(hXt->getFormatType(0)) {
    case INDICATOR :
        computeXBetaImpl<IndicatorIterator<RealType>>(beta);
        break;
    case SPARSE :
        computeXBetaImpl<SparseIterator<RealType>>(beta);
        break;
    case DENSE :
        computeXBetaImpl<DenseIterator<RealType>>(beta);
        break;
    case INTERCEPT:
        break;
}


#ifdef CYCLOPS_DEBUG_TIMING
auto end = bsccs::chrono::steady_clock::now();
///////////////////////////"
duration["computeXBeta     "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

}

template <class BaseModel,typename RealType> template <class IteratorType>
void ModelSpecifics<BaseModel,RealType>::computeXBetaImpl(double *beta) {

    for (int k = 0; k < K; ++k) {
        RealType sum = 0.0;
        IteratorType it(*hXt, k);
        for (; it; ++it) {
            const auto j = it.index();
            sum += it.value() * beta[j];
        }
        hXBeta[k] = sum;
        // TODO Add back in for fastest LR
//         const auto exb = std::exp(sum);
//         offsExpXBeta[k] = exb;
//         denomPid[k] = 1.0 + exb;
    }

    // std::cerr << "did weird stuff to denomPid" << std::endl;
}

template <class BaseModel,typename RealType>
bool ModelSpecifics<BaseModel,RealType>::allocateXjY(void) { return BaseModel::precomputeGradient; }

template <class BaseModel,typename RealType>
bool ModelSpecifics<BaseModel,RealType>::allocateXjX(void) { return BaseModel::precomputeHessian; }

template <class BaseModel,typename RealType>
bool ModelSpecifics<BaseModel,RealType>::sortPid(void) { return BaseModel::sortPid; }

template <class BaseModel,typename RealType>
bool ModelSpecifics<BaseModel,RealType>::initializeAccumulationVectors(void) { return BaseModel::cumulativeGradientAndHessian; }

template <class BaseModel,typename RealType>
bool ModelSpecifics<BaseModel,RealType>::allocateNtoKIndices(void) { return BaseModel::hasNtoKIndices; }

template <class BaseModel,typename RealType>
bool ModelSpecifics<BaseModel,RealType>::hasResetableAccumulators(void) { return BaseModel::hasResetableAccumulators; }

template <class BaseModel,typename RealType> template <class IteratorType>
void ModelSpecifics<BaseModel,RealType>::axpy(RealType* y, const RealType alpha, const int index) {
	IteratorType it(hX, index);
	for (; it; ++it) {
		const int k = it.index();
		y[k] += alpha * it.value();
	}
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::axpyXBeta(const double beta, const int j) {
	syncSparseLogisticState();
	sparseLogisticStateReady = false;

#ifdef CYCLOPS_DEBUG_TIMING
    auto start = bsccs::chrono::steady_clock::now();
#endif

	if (beta != 0.0) {
		switch (hX.getFormatType(j)) {
		case INDICATOR:
			axpy < IndicatorIterator<RealType> > (hXBeta.data(), beta, j);
			break;
		case INTERCEPT:
		    axpy < InterceptIterator<RealType> > (hXBeta.data(), beta, j);
		    break;
		case DENSE:
			axpy < DenseIterator<RealType> > (hXBeta.data(), beta, j);
			break;
		case SPARSE:
			axpy < SparseIterator<RealType> > (hXBeta.data(), beta, j);
			break;
		}
	}

#ifdef CYCLOPS_DEBUG_TIMING
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["axpy             "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

}


// ESK: Added cWeights (censoring weights) as an input
template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::setWeights(double* inWeights, double *cenWeights, bool useCrossValidation) {
	syncSparseLogisticState();
	sparseLogisticStateReady = false;
	invalidateSparseLogisticWeightedColumnViews();

	// Set K weights
	if (hKWeight.size() != K) {
		hKWeight.resize(K);
	}

	if (useCrossValidation) {
		for (size_t k = 0; k < K; ++k) {
			hKWeight[k] = inWeights[k];
		}
	} else {
		std::fill(hKWeight.begin(), hKWeight.end(), static_cast<RealType>(1));
    }

	if (initializeAccumulationVectors()) {
		setPidForAccumulation(inWeights);
	}

	// Set N weights (these are the same for independent data models)
	if (hNWeight.size() < N + 1) { // Add +1 for extra (zero-weight stratum)
		hNWeight.resize(N + 1);
	}

	std::fill(hNWeight.begin(), hNWeight.end(), static_cast<RealType>(0));
    for (size_t k = 0; k < K; ++k) {
        RealType event = BaseModel::observationCount(hY[k]) * hKWeight[k];
        incrementByGroup(hNWeight.data(), hPid, k, event);
    }

    //ESK: Compute hNtoK indices manually and create censoring weights hYWeight is isTwoWayScan
    if (hYWeight.size() != K) {
        hYWeight.resize(K);
    }
	if (hYWeightDouble.size() != K) {
		hYWeightDouble.resize(K);
	}
    if (BaseModel::isTwoWayScan) {
        hNtoK.resize(N + 1);
        int n = 0;
        for (size_t k = 0; k < K;) {
            while (hKWeight[k] == static_cast<RealType>(0)) {
                ++k;
            }
            hNtoK[n] = k;
            int currentPid = hPid[k];
            do {
                ++k;
            } while (k < K && (currentPid == hPid[k] || hKWeight[k] == static_cast<RealType>(0)));
            ++n;
        }
        hNtoK[n] = K;

        for (size_t k = 0; k < K; ++k) {
            hYWeight[k] = cenWeights[k];
            hYWeightDouble[k] = cenWeights[k];
        }
    }

#ifdef DEBUG_COX
	cerr << "Done with set weights" << endl;
#endif

}

template<class BaseModel, typename RealType>
void ModelSpecifics<BaseModel, RealType>::computeXjY(bool useCrossValidation) {
	for (size_t j = 0; j < J; ++j) {
		hXjY[j] = 0;

		GenericIterator<RealType> it(hX, j);

		if (useCrossValidation) {
			for (; it; ++it) {
                const int k = it.index();
				if (BaseModel::exactTies && hNWeight[BaseModel::getGroup(hPid, k)] > 1) {
					// Do not precompute
					hXjY[j] += it.value() * hY[k] * hKWeight[k]; // TODO Remove
				} else if (BaseModel::isSurvivalModel) {
				    // ESK: Modified for suvival data
                    hXjY[j] += it.value() * BaseModel::observationCount(hY[k]) * hKWeight[k];
				} else {
					hXjY[j] += it.value() * hY[k] * hKWeight[k];
				}
			}
		} else {
			for (; it; ++it) {
                const int k = it.index();
				if (BaseModel::exactTies && hNWeight[BaseModel::getGroup(hPid, k)] > 1) {
					// Do not precompute
                    hXjY[j] += it.value() * hY[k];
				} else if (BaseModel::isSurvivalModel) {
                    // ESK: Modified for survival data
                    hXjY[j] += it.value() * BaseModel::observationCount(hY[k]);
				} else {
				    hXjY[j] += it.value() * hY[k];
                }
			}
		}
#ifdef DEBUG_COX
		cerr << "j: " << j << " = " << hXjY[j]<< endl;
#endif
	}
}

template<class BaseModel, typename RealType>
void ModelSpecifics<BaseModel, RealType>::computeXjX(bool useCrossValidation) {
	for (size_t j = 0; j < J; ++j) {
		hXjX[j] = 0;
		GenericIterator<RealType> it(hX, j);

		if (useCrossValidation) {
			for (; it; ++it) {
				const int k = it.index();
				if (BaseModel::exactTies && hNWeight[BaseModel::getGroup(hPid, k)] > 1) {
					// Do not precompute
				} else {
					hXjX[j] += it.value() * it.value() * hKWeight[k];
				}
			}
		} else {
			for (; it; ++it) {
				const int k = it.index();
				if (BaseModel::exactTies && hNWeight[BaseModel::getGroup(hPid, k)] > 1) {
					// Do not precompute
				} else {
					hXjX[j] += it.value() * it.value();
				}
			}
		}
	}
}

template<class BaseModel, typename RealType>
void ModelSpecifics<BaseModel, RealType>::computeNtoKIndices(bool useCrossValidation) {

	hNtoK.resize(N+1);
	int n = 0;
	for (size_t k = 0; k < K;) {
		hNtoK[n] = k;
		int currentPid = hPid[k];
		do {
			++k;
		} while (k < K && currentPid == hPid[k]);
		++n;
	}
	hNtoK[n] = K;
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeFixedTermsInLogLikelihood(bool useCrossValidation) {
	if(BaseModel::likelihoodHasFixedTerms) {
		logLikelihoodFixedTerm = static_cast<RealType>(0);
	    bool hasOffs = hOffs.size() > 0;
		if (useCrossValidation) {
			for(size_t i = 0; i < K; i++) {
			    auto offs = hasOffs ? hOffs[i] : static_cast<RealType>(0);
				logLikelihoodFixedTerm += BaseModel::logLikeFixedTermsContrib(hY[i], offs, offs) * hKWeight[i];
			}
		} else {
			for(size_t i = 0; i < K; i++) {
			    auto offs = hasOffs ? hOffs[i] : static_cast<RealType>(0);
				logLikelihoodFixedTerm += BaseModel::logLikeFixedTermsContrib(hY[i], offs, offs); // TODO SEGV in Poisson model
			}
		}
	}
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeFixedTermsInGradientAndHessian(bool useCrossValidation) {
#ifdef CYCLOPS_DEBUG_TIMING
    auto start = bsccs::chrono::steady_clock::now();
#endif
	if (sortPid()) {
		doSortPid(useCrossValidation);
	}
	if (allocateXjY()) {
		computeXjY(useCrossValidation);
	}
	if (allocateXjX()) {
		computeXjX(useCrossValidation);
	}
	if (allocateNtoKIndices()) {
		computeNtoKIndices(useCrossValidation);
	}
#ifdef CYCLOPS_DEBUG_TIMING
    auto end = bsccs::chrono::steady_clock::now();
    ///////////////////////////"
    auto name = "compFixedGH      ";
    duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
}

template <class BaseModel,typename RealType>
double ModelSpecifics<BaseModel,RealType>::getLogLikelihood(bool useCrossValidation) {

#ifdef CYCLOPS_DEBUG_TIMING
	auto start = bsccs::chrono::steady_clock::now();
#endif
	RealType logLikelihood = static_cast<RealType>(0.0);
	if (sparseLogisticXBetaDirty && std::is_same<BaseModel, LogisticRegression<RealType>>::value) {
		if (useCrossValidation) {
			for (const int i : sparseLogisticCaseRows) {
				if (hKWeight[i] != static_cast<RealType>(0.0)) {
					logLikelihood += sparseLogisticLogOdds(sparseLogisticFitted[i]) * hKWeight[i];
				}
			}
		} else {
			for (const int i : sparseLogisticCaseRows) {
				logLikelihood += sparseLogisticLogOdds(sparseLogisticFitted[i]);
			}
		}
	} else if (useCrossValidation) {
		for (size_t i = 0; i < K; i++) {
			logLikelihood += BaseModel::logLikeNumeratorContrib(hY[i], hXBeta[i]) * hKWeight[i];
		}
	} else {
		for (size_t i = 0; i < K; i++) {
            logLikelihood += BaseModel::logLikeNumeratorContrib(hY[i], hXBeta[i]);
		}
    }

    if (BaseModel::likelihoodHasDenominator) { // Compile-time switch
		if(BaseModel::cumulativeGradientAndHessian) {
			for (size_t i = 0; i < N; i++) {
				// Weights modified in computeNEvents()
				logLikelihood -= BaseModel::logLikeDenominatorContrib(hNWeight[i], accDenomPid[i]);
			}
		} else {  // TODO Unnecessary code duplication
			for (size_t i = 0; i < N; i++) {
				// Weights modified in computeNEvents()
				if (sparseLogisticXBetaDirty && std::is_same<BaseModel, LogisticRegression<RealType>>::value) {
					logLikelihood -= hNWeight[i] * sparseLogisticLogDenominator(sparseLogisticFitted[i]);
				} else {
					logLikelihood -= BaseModel::logLikeDenominatorContrib(hNWeight[i], denomPid[i]);
				}
			}
		}
	}

    // RANGE

	if (BaseModel::likelihoodHasFixedTerms) {
		logLikelihood += logLikelihoodFixedTerm;
	}

#ifdef CYCLOPS_DEBUG_TIMING
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["compLogLike      "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

	return static_cast<double>(logLikelihood);
}

template <class BaseModel,typename RealType>
double ModelSpecifics<BaseModel,RealType>::getPredictiveLogLikelihood(double* weights) {
	std::vector<double> saveKWeight;
	if (BaseModel::cumulativeGradientAndHessian)	{

 	    // saveKWeight = hKWeight; // make copy
	    if (saveKWeight.size() != K) {
	        saveKWeight.resize(K);
	    }
	    for (size_t k = 0; k < K; ++k) {
	        saveKWeight[k] = hKWeight[k]; // make copy to a double vector
	    }

		setPidForAccumulation(weights);
		setWeights(weights, BaseModel::isTwoWayScan ? hYWeightDouble.data() : nullptr, true); // set new weights // TODO Possible error for gfr
		computeRemainingStatistics(true); // compute accDenomPid
	}

	RealType logLikelihood = static_cast<RealType>(0.0);

	if (BaseModel::cumulativeGradientAndHessian) {
	    for (size_t k = 0; k < K; ++k) {
	        logLikelihood += BaseModel::logPredLikeContrib(hY[k], weights[k], hXBeta[k], &accDenomPid[0], hPid, k); // TODO Going to crash with ties
	    }
	} else if (sparseLogisticXBetaDirty && std::is_same<BaseModel, LogisticRegression<RealType>>::value) {
		for (const int k : sparseLogisticCaseRows) {
			if (weights[k] != static_cast<RealType>(0.0)) {
				logLikelihood += weights[k] *
					sparseLogisticLogOdds(sparseLogisticFitted[k]);
			}
		}
		for (size_t k = 0; k < K; ++k) {
			if (weights[k] != static_cast<RealType>(0.0)) {
				logLikelihood -= weights[k] *
					sparseLogisticLogDenominator(sparseLogisticFitted[k]);
			}
		}
	} else { // TODO Unnecessary code duplication
	    for (size_t k = 0; k < K; ++k) { // TODO Is index of K correct?
	        logLikelihood += BaseModel::logPredLikeContrib(hY[k], weights[k], hXBeta[k], &denomPid[0], hPid, k);
	    }
	}

	if (BaseModel::cumulativeGradientAndHessian) {
		setPidForAccumulation(&saveKWeight[0]);
		setWeights(saveKWeight.data(), BaseModel::isTwoWayScan ? hYWeightDouble.data() : nullptr, true); // set old weights // TODO Possible error for gfr
		computeRemainingStatistics(true);
	}

	return static_cast<double>(logLikelihood);
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::getPredictiveEstimates(double* y, double* weights){
	if (weights) {
		for (size_t k = 0; k < K; ++k) {
			if (weights[k]) {
				y[k] = sparseLogisticXBetaDirty && std::is_same<BaseModel, LogisticRegression<RealType>>::value
					? sparseLogisticFitted[k]
					: BaseModel::predictEstimate(hXBeta[k]);
			}
		}
	} else {
		for (size_t k = 0; k < K; ++k) {
			y[k] = sparseLogisticXBetaDirty && std::is_same<BaseModel, LogisticRegression<RealType>>::value
				? sparseLogisticFitted[k]
				: BaseModel::predictEstimate(hXBeta[k]);
		}
	}
	// TODO How to remove code duplication above?
}


template <class RealType>
void makeDenseCrossProduct(const CompressedDataMatrix<RealType>& x,
                                                int index1, int index2) {

}

template <class BaseModel, typename RealType> template <class IteratorType, class Weights>
void ModelSpecifics<BaseModel,RealType>::getSchoenfeldResidualsImpl(int index,
                                                                    std::vector<double>* residuals,
                                                                    std::vector<double>* times,
                                                                    std::vector<int>* strata,
                                                                    double* covariate,
                                                                    double* score,
                                                                    Weights w) {

    const bool hasResiduals = residuals != nullptr;
    const bool hasTimes = times != nullptr;
    const bool hasStrata = strata != nullptr;
    const bool hasScore = score != nullptr && covariate != nullptr;

    if (hasResiduals) {
        residuals->clear();
    }
    if (hasTimes) {
        times->clear();
    }
    if (hasStrata) {
        strata->clear();
    }

    RealType uGradient = static_cast<RealType>(0); // unweighted
    RealType uHessian = static_cast<RealType>(0);
    RealType wGradient = static_cast<RealType>(0); // weighted
    RealType wHessian = static_cast<RealType>(0);
    RealType xHessian = static_cast<RealType>(0);

    RealType uGradient2 = static_cast<RealType>(0);
    RealType wGradient2 = static_cast<RealType>(0);

    // TODO: only written for accummulive models (Cox, Fine/Grey)

	IteratorType it(hX, index);

	RealType resNumerator  = static_cast<RealType>(0);
	// RealType resDenominator = static_cast<RealType>(0);
	RealType scoreNumerator1 = static_cast<RealType>(0);
	RealType scoreNumerator2 = static_cast<RealType>(0);
	RealType scoreDenominator = static_cast<RealType>(0);

	// find start relavent accumulator reset point
	auto reset = begin(accReset);
	// while( *reset < it.index() ) {
	// 	++reset;
	// }

	std::deque<RealType> deferredResidualX;
	std::deque<RealType> deferredScore1;
	std::deque<RealType> deferredScore2;
	std::deque<RealType> deferredScore3;
	std::deque<int> deferredScore4;

	auto processRow = [&](int i, RealType x) {

	    if (*reset <= hPid[i]) {

	        resNumerator = static_cast<RealType>(0);
	        // resDenominator = static_cast<RealType>(0);
	        scoreNumerator1 = static_cast<RealType>(0);
	        scoreNumerator2 = static_cast<RealType>(0);
	        scoreDenominator = static_cast<RealType>(0);

	        ++reset;
	    }

		const auto expXBeta = offsExpXBeta[i]; // std::exp(hXBeta[i]);

		resNumerator += hKWeight[i] * expXBeta * x;
		// resDenominator += expXBeta;

		if (hY[i] == 1) {
		    // std::cerr << " " << x << " for " << resNumerator << " / " << resDenominator << " - " <<  ( x - resNumerator / resDenominator) << "\n";
		    // std::cerr << denomPid[i] << "\n";
			if (hasResiduals) {
			    if (i < (K - 1) && hPid[i] < *reset &&
                    (hOffs[i] == hOffs[i+ 1]) && hY[i + 1] == 1) {
			        deferredResidualX.push_back(x);
			    } else {

			        const auto denom = accDenomPid[hPid[i]];

			        while (deferredResidualX.size() > 0) {
			            const auto residual = deferredResidualX.front() - resNumerator / denom;
			            residuals->push_back(residual);
			            deferredResidualX.pop_front();
			        }

			        const auto residual = x - resNumerator / denom;
				    residuals->push_back(residual);
			    }
			}
			if (hasTimes) {
				times->push_back(hOffs[i]);
			}
			if (hasStrata) {
				strata->push_back(hPidOriginal[i]);
			}
		} else {
		    // std::cerr << "\n";
		}

		if (hasScore) {
			const auto cov = covariate[i];
			const auto numerator1 = hKWeight[i] * expXBeta * x;
			const auto numerator2 = hKWeight[i] * expXBeta * x * x;

			if (hY[i] == 1) {
				uGradient2 += hKWeight[i] * x;
				wGradient2 += hKWeight[i] * x * cov;
			}

			scoreNumerator1 += numerator1;
			scoreNumerator2 += numerator2;

		    if (i < (K - 1) && hPid[i] == hPid[i + 1]) {
                // zero contribution under Breslow
		    } else {

			    const auto denom = accDenomPid[hPid[i]];

	    		const auto t = scoreNumerator1 / denom;
			    const auto gradient = hNWeight[hPid[i]] * t;
			    const auto hessian = hNWeight[hPid[i]] * (scoreNumerator2 / denom - t * t);

			    uGradient += gradient;
			    wGradient += gradient * cov;

			    // std::cerr << "newA" << i << " " << hPid[i] << " " << scoreNumerator1 << "/" << denom << " " << hY[i] << " " << uGradient << " " << hNWeight[hPid[i]] << "\n";

			    uHessian += hessian;
			    wHessian += hessian * cov * cov;
			    xHessian += hessian * cov;
		    }
		}
	};

	// main loop
	for (int i = 0; i < K; ++i) {
	    if (i == it.index()) {
	        processRow(i, it.value());
	        ++it;
	    } else {
	        processRow(i, static_cast<RealType>(0));
	    }
	}

    if (hasScore) {

        score[0] = static_cast<double>(uGradient2 - uGradient);
        score[1] = static_cast<double>(wGradient2 - wGradient);
        score[2] = static_cast<double>(uHessian);
        score[3] = static_cast<double>(xHessian);
        score[4] = static_cast<double>(xHessian);
        score[5] = static_cast<double>(wHessian);
     }
}

template <class BaseModel,typename RealType>
bool ModelSpecifics<BaseModel,RealType>::supportsSparseLogisticKernel(int index, bool useWeights) const {
	return std::is_same<BaseModel, LogisticRegression<RealType>>::value
		&& index >= 0
		&& static_cast<size_t>(index) < J;
}

template <class BaseModel,typename RealType>
size_t ModelSpecifics<BaseModel,RealType>::getSparseLogisticColumnScanSize(int index, bool useWeights) {
	if (!supportsSparseLogisticKernel(index, useWeights)) {
		return hX.getNumberOfEntries(index);
	}
	ensureSparseLogisticColumnFormat();
	if (useWeights) {
		ensureSparseLogisticWeightedColumnViews();
		switch (sparseLogisticColumnFormat[index]) {
			case INDICATOR:
			case SPARSE:
				return sparseLogisticWeightedColumnViews[index].rows.size();
			case DENSE:
			case INTERCEPT:
				return sparseLogisticWeightedRows.size();
		}
	}
	switch (sparseLogisticColumnFormat[index]) {
		case INDICATOR:
		case SPARSE:
			return hX.getNumberOfEntries(index);
		case DENSE:
		case INTERCEPT:
			return K;
	}
	return 0;
}

template <class BaseModel,typename RealType>
SparseLogisticPerformanceCounters ModelSpecifics<BaseModel,RealType>::getSparseLogisticPerformanceCounters() const {
	return sparseLogisticPerformanceCounters;
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::resetSparseLogisticPerformanceCounters() {
	sparseLogisticPerformanceCounters = SparseLogisticPerformanceCounters();
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::addSparseLogisticKktScan(size_t nnz, double seconds) {
	sparseLogisticPerformanceCounters.fullKktScans += 1;
	sparseLogisticPerformanceCounters.kktGradientCalls += 1;
	sparseLogisticPerformanceCounters.nnzKkt += nnz;
	sparseLogisticPerformanceCounters.kktSeconds += seconds;
}

template <class BaseModel,typename RealType>
GapSafeDiagnostic ModelSpecifics<BaseModel,RealType>::computeSparseLogisticGapSafeDiagnostic(
		const std::vector<double>& boundaries,
		const std::vector<bool>& fixedBeta,
		const std::vector<double>& beta,
		double penalty,
		bool useWeights) {
	GapSafeDiagnostic diagnostic;
	const auto start = std::chrono::steady_clock::now();
	if (!std::is_same<BaseModel, LogisticRegression<RealType>>::value
			|| boundaries.size() != static_cast<size_t>(J)
			|| fixedBeta.size() != static_cast<size_t>(J)
			|| beta.size() != static_cast<size_t>(J)) {
		return diagnostic;
	}
	refreshSparseLogisticState(useWeights);
	ensureSparseLogisticColumnFormat();
	if (useWeights) {
		ensureSparseLogisticWeightedColumnViews();
	}

	const RealType one = static_cast<RealType>(1.0);
	const RealType zero = static_cast<RealType>(0.0);
	const RealType eps = std::numeric_limits<RealType>::epsilon();
	const RealType* fitted = sparseLogisticFitted.data();
	const RealType* weights = hKWeight.data();
	auto clampProbability = [eps, one](RealType p) {
		if (p < eps) {
			return eps;
		}
		const RealType upper = one - eps;
		if (p > upper) {
			return upper;
		}
		return p;
	};
	auto entropyTerm = [](const long double q) {
		if (q <= 0.0L || q >= 1.0L) {
			return 0.0L;
		}
		return q * std::log(q) + (1.0L - q) * std::log(1.0L - q);
	};

	long double loss = 0.0L;
	long double weightedCases = 0.0L;
	long double totalWeight = 0.0L;
	RealType maxWeight = zero;
	for (size_t i = 0; i < K; ++i) {
		const RealType w = useWeights ? weights[i] : one;
		if (w == zero) {
			continue;
		}
		maxWeight = std::max(maxWeight, w);
		totalWeight += static_cast<long double>(w);
		weightedCases += static_cast<long double>(w) * static_cast<long double>(hY[i]);
		const RealType p = clampProbability(fitted[i]);
		const RealType y = hY[i];
		loss -= static_cast<long double>(w)
			* (static_cast<long double>(y) * std::log(static_cast<long double>(p))
					+ static_cast<long double>(one - y)
						* std::log1p(-static_cast<long double>(p)));
	}
	diagnostic.primalObjective = static_cast<double>(loss + static_cast<long double>(penalty));

	std::vector<RealType> dualProbability(K, zero);
	if (totalWeight > 0.0L) {
		auto shiftedCaseTotal = [&](const long double shift) {
			long double total = 0.0L;
			for (size_t i = 0; i < K; ++i) {
				const RealType w = useWeights ? weights[i] : one;
				if (w == zero) {
					continue;
				}
				const RealType p = clampProbability(fitted[i]);
				const long double logit =
					std::log(static_cast<long double>(p))
					- std::log1p(-static_cast<long double>(p));
				const long double eta = logit + shift;
				const long double q = eta >= 0.0L
					? 1.0L / (1.0L + std::exp(-eta))
					: std::exp(eta) / (1.0L + std::exp(eta));
				total += static_cast<long double>(w) * q;
			}
			return total;
		};
		long double lower = -50.0L;
		long double upper = 50.0L;
		for (int expand = 0; expand < 4 && shiftedCaseTotal(lower) > weightedCases; ++expand) {
			lower *= 2.0L;
		}
		for (int expand = 0; expand < 4 && shiftedCaseTotal(upper) < weightedCases; ++expand) {
			upper *= 2.0L;
		}
		for (int iter = 0; iter < 80; ++iter) {
			const long double mid = (lower + upper) / 2.0L;
			if (shiftedCaseTotal(mid) < weightedCases) {
				lower = mid;
			} else {
				upper = mid;
			}
		}
		const long double shift = (lower + upper) / 2.0L;
		for (size_t i = 0; i < K; ++i) {
			const RealType w = useWeights ? weights[i] : one;
			if (w == zero) {
				dualProbability[i] = hY[i];
				continue;
			}
			const RealType p = clampProbability(fitted[i]);
			const long double logit =
				std::log(static_cast<long double>(p))
				- std::log1p(-static_cast<long double>(p));
			const long double eta = logit + shift;
			const long double q = eta >= 0.0L
				? 1.0L / (1.0L + std::exp(-eta))
				: std::exp(eta) / (1.0L + std::exp(eta));
			dualProbability[i] = static_cast<RealType>(q);
		}
	}

	struct ColumnDiagnostic {
		double gradient = 0.0;
		double norm2 = 0.0;
		double sum = 0.0;
		size_t nnz = 0;
	};
	auto scanColumn = [&](const int j) {
		ColumnDiagnostic column;
		long double gradient = 0.0L;
		long double norm2 = 0.0L;
		long double sum = 0.0L;
		switch (sparseLogisticColumnFormat[j]) {
			case INDICATOR: {
				const SparseLogisticWeightedColumnView* view =
					useWeights ? &sparseLogisticWeightedColumnViews[j] : nullptr;
				const int* rows = useWeights ? view->rows.data() : hX.getCompressedColumnVector(j);
				const size_t n = useWeights ? view->rows.size() : hX.getNumberOfEntries(j);
				column.nnz = n;
				for (size_t row = 0; row < n; ++row) {
					const int i = rows[row];
					const RealType w = (useWeights && !sparseLogisticWeightedRowsHaveUnitWeight)
						? weights[i]
						: one;
					gradient += static_cast<long double>(w) * dualProbability[i];
					norm2 += 1.0L;
					sum += 1.0L;
				}
				break;
			}
			case SPARSE: {
				const SparseLogisticWeightedColumnView* view =
					useWeights ? &sparseLogisticWeightedColumnViews[j] : nullptr;
				const int* rows = useWeights ? view->rows.data() : hX.getCompressedColumnVector(j);
				const RealType* values = useWeights ? view->values.data() : hX.getDataVector(j);
				const size_t n = useWeights ? view->rows.size() : hX.getNumberOfEntries(j);
				column.nnz = n;
				for (size_t row = 0; row < n; ++row) {
					const int i = rows[row];
					const RealType x = values[row];
					const RealType w = (useWeights && !sparseLogisticWeightedRowsHaveUnitWeight)
						? weights[i]
						: one;
					gradient += static_cast<long double>(w) * x * dualProbability[i];
					norm2 += static_cast<long double>(x) * x;
					sum += static_cast<long double>(x);
				}
				break;
			}
			case DENSE: {
				const RealType* values = hX.getDataVector(j);
				const size_t n = useWeights ? sparseLogisticWeightedRows.size() : K;
				column.nnz = n;
				for (size_t row = 0; row < n; ++row) {
					const int i = useWeights ? sparseLogisticWeightedRows[row] : static_cast<int>(row);
					const RealType x = values[i];
					const RealType w = (useWeights && !sparseLogisticWeightedRowsHaveUnitWeight)
						? weights[i]
						: one;
					gradient += static_cast<long double>(w) * x * dualProbability[i];
					norm2 += static_cast<long double>(x) * x;
					sum += static_cast<long double>(x);
				}
				break;
			}
			case INTERCEPT: {
				const size_t n = useWeights ? sparseLogisticWeightedRows.size() : K;
				column.nnz = n;
				for (size_t row = 0; row < n; ++row) {
					const int i = useWeights ? sparseLogisticWeightedRows[row] : static_cast<int>(row);
					const RealType w = (useWeights && !sparseLogisticWeightedRowsHaveUnitWeight)
						? weights[i]
						: one;
					gradient += static_cast<long double>(w) * dualProbability[i];
					norm2 += 1.0L;
					sum += 1.0L;
				}
				break;
			}
		}
		gradient -= hXjY[j];
		column.gradient = static_cast<double>(gradient);
		column.norm2 = static_cast<double>(norm2);
		column.sum = static_cast<double>(sum);
		return column;
	};

	std::vector<ColumnDiagnostic> columns(J);
	double maxGradientRatio = 0.0;
	double maxUnpenalizedGradient = 0.0;
	for (int j = 0; j < J; ++j) {
		columns[j] = scanColumn(j);
		diagnostic.totalColumns += fixedBeta[j] ? 0 : 1;
		diagnostic.totalNnz += fixedBeta[j] ? 0 : columns[j].nnz;
		if (fixedBeta[j]) {
			continue;
		}
		const double boundary = boundaries[j];
		if (boundary > 0.0) {
			maxGradientRatio = std::max(maxGradientRatio,
					std::abs(columns[j].gradient) / boundary);
			if (beta[j] == 0.0) {
				diagnostic.eligibleColumns += 1;
				diagnostic.eligibleNnz += columns[j].nnz;
			}
		} else {
			maxUnpenalizedGradient = std::max(maxUnpenalizedGradient,
					std::abs(columns[j].gradient));
		}
	}

	const char* unpenalizedToleranceEnv = std::getenv("CYCLOPS_GAP_SAFE_UNPENALIZED_TOLERANCE");
	const double unpenalizedTolerance = unpenalizedToleranceEnv != nullptr
		? std::atof(unpenalizedToleranceEnv)
		: 1e-6;
	diagnostic.maxGradientRatio = maxGradientRatio;
	diagnostic.maxUnpenalizedGradient = maxUnpenalizedGradient;
	diagnostic.dualFeasible = maxUnpenalizedGradient <= unpenalizedTolerance;
	diagnostic.dualScale = diagnostic.dualFeasible
		? (maxGradientRatio > 1.0 ? 1.0 / maxGradientRatio : 1.0)
		: 0.0;

	long double conjugate = 0.0L;
	for (size_t i = 0; i < K; ++i) {
		const RealType w = useWeights ? weights[i] : one;
		if (w == zero) {
			continue;
		}
		long double q = static_cast<long double>(hY[i])
			+ static_cast<long double>(diagnostic.dualScale)
				* (static_cast<long double>(dualProbability[i]) - static_cast<long double>(hY[i]));
		if (q < 0.0L) {
			q = 0.0L;
		} else if (q > 1.0L) {
			q = 1.0L;
		}
		conjugate += static_cast<long double>(w) * entropyTerm(q);
	}
	diagnostic.dualObjective = static_cast<double>(-conjugate);
	diagnostic.dualityGap = diagnostic.primalObjective - diagnostic.dualObjective;
	if (diagnostic.dualityGap < 0.0 && diagnostic.dualityGap > -1e-8) {
		diagnostic.dualityGap = 0.0;
	}
	if (diagnostic.dualityGap < 0.0) {
		diagnostic.dualFeasible = false;
		diagnostic.dualScale = 0.0;
	}
	if (maxWeight > zero && diagnostic.dualityGap >= 0.0) {
		diagnostic.safeRadius = std::sqrt(diagnostic.dualityGap
				* static_cast<double>(maxWeight) / 2.0);
	}
	if (diagnostic.dualFeasible) {
		for (int j = 0; j < J; ++j) {
			if (fixedBeta[j] || boundaries[j] <= 0.0 || beta[j] != 0.0) {
				continue;
			}
			const double dualCorrelation =
				std::abs(diagnostic.dualScale * columns[j].gradient);
			const double safeBound =
				dualCorrelation + std::sqrt(columns[j].norm2) * diagnostic.safeRadius;
			if (safeBound < boundaries[j]) {
				diagnostic.screenedColumns += 1;
				diagnostic.screenedNnz += columns[j].nnz;
			}
		}
	}

	const char* radiusTracePath = std::getenv("CYCLOPS_GAP_SAFE_RADIUS_TRACE");
	if (diagnostic.dualFeasible && radiusTracePath != nullptr && radiusTracePath[0] != '\0') {
		struct RadiusBucketCounts {
			uint64_t columns = 0;
			uint64_t nnz = 0;
			uint64_t rareNnz = 0;
			uint64_t smallNnz = 0;
			uint64_t mediumNnz = 0;
			uint64_t hotNnz = 0;
		};
		auto addBucket = [](RadiusBucketCounts& counts, const size_t nnz) {
			counts.columns += 1;
			counts.nnz += nnz;
			if (nnz < 32) {
				counts.rareNnz += nnz;
			} else if (nnz < 256) {
				counts.smallNnz += nnz;
			} else if (nnz < 4096) {
				counts.mediumNnz += nnz;
			} else {
				counts.hotNnz += nnz;
			}
		};
		const double radiusDivisors[] = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 0.0};
		std::vector<RadiusBucketCounts> uncenteredCounts(8);
		std::vector<RadiusBucketCounts> centeredCounts(8);
		uint64_t eligibleRareNnz = 0;
		uint64_t eligibleSmallNnz = 0;
		uint64_t eligibleMediumNnz = 0;
		uint64_t eligibleHotNnz = 0;
		for (int j = 0; j < J; ++j) {
			if (fixedBeta[j] || boundaries[j] <= 0.0 || beta[j] != 0.0) {
				continue;
			}
			if (columns[j].nnz < 32) {
				eligibleRareNnz += columns[j].nnz;
			} else if (columns[j].nnz < 256) {
				eligibleSmallNnz += columns[j].nnz;
			} else if (columns[j].nnz < 4096) {
				eligibleMediumNnz += columns[j].nnz;
			} else {
				eligibleHotNnz += columns[j].nnz;
			}
			const double dualCorrelation =
				std::abs(diagnostic.dualScale * columns[j].gradient);
			const double centeredNorm2 = totalWeight > 0.0L
				? std::max(0.0, columns[j].norm2
						- (columns[j].sum * columns[j].sum / static_cast<double>(totalWeight)))
				: columns[j].norm2;
			for (size_t r = 0; r < 8; ++r) {
				const double radius = radiusDivisors[r] == 0.0
					? 0.0
					: diagnostic.safeRadius / radiusDivisors[r];
				const double uncenteredSafeBound =
					dualCorrelation + std::sqrt(columns[j].norm2) * radius;
				const double centeredSafeBound =
					dualCorrelation + std::sqrt(centeredNorm2) * radius;
				if (uncenteredSafeBound < boundaries[j]) {
					addBucket(uncenteredCounts[r], columns[j].nnz);
				}
				if (centeredSafeBound < boundaries[j]) {
					addBucket(centeredCounts[r], columns[j].nnz);
				}
			}
		}

		static std::mutex radiusTraceMutex;
		std::lock_guard<std::mutex> lock(radiusTraceMutex);
		std::ifstream existing(radiusTracePath);
		const bool writeHeader = !existing.good() || existing.peek() == std::ifstream::traits_type::eof();
		existing.close();
		std::ofstream out(radiusTracePath, std::ios::app);
		if (writeHeader) {
			out << "centered,radius_divisor,implied_gap_factor,radius,"
				<< "primal_objective,dual_objective,duality_gap,dual_scale,"
				<< "eligible_columns,eligible_nnz,eligible_rare_nnz,eligible_small_nnz,"
				<< "eligible_medium_nnz,eligible_hot_nnz,"
				<< "screened_columns,screened_nnz,screened_rare_nnz,screened_small_nnz,"
				<< "screened_medium_nnz,screened_hot_nnz\n";
		}
		out.precision(17);
		for (size_t r = 0; r < 8; ++r) {
			const double divisor = radiusDivisors[r];
			const double radius = divisor == 0.0 ? 0.0 : diagnostic.safeRadius / divisor;
			const double impliedGapFactor = divisor == 0.0 ? 0.0 : 1.0 / (divisor * divisor);
			for (int centered = 0; centered <= 1; ++centered) {
				const auto& counts = centered == 0 ? uncenteredCounts[r] : centeredCounts[r];
				out << centered << ','
					<< divisor << ','
					<< impliedGapFactor << ','
					<< radius << ','
					<< diagnostic.primalObjective << ','
					<< diagnostic.dualObjective << ','
					<< diagnostic.dualityGap << ','
					<< diagnostic.dualScale << ','
					<< diagnostic.eligibleColumns << ','
					<< diagnostic.eligibleNnz << ','
					<< eligibleRareNnz << ','
					<< eligibleSmallNnz << ','
					<< eligibleMediumNnz << ','
					<< eligibleHotNnz << ','
					<< counts.columns << ','
					<< counts.nnz << ','
					<< counts.rareNnz << ','
					<< counts.smallNnz << ','
					<< counts.mediumNnz << ','
					<< counts.hotNnz << '\n';
			}
		}
	}

	const auto end = std::chrono::steady_clock::now();
	diagnostic.available = true;
	diagnostic.seconds = std::chrono::duration<double>(end - start).count();
	return diagnostic;
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::ensureSparseLogisticColumnFormat() {
	const size_t columns = hX.getNumberOfColumns();
	if (sparseLogisticColumnFormat.size() == columns) {
		return;
	}
	sparseLogisticColumnFormat.resize(columns);
	for (size_t j = 0; j < columns; ++j) {
		sparseLogisticColumnFormat[j] = hX.getFormatType(static_cast<int>(j));
	}
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::invalidateSparseLogisticWeightedColumnViews() {
	sparseLogisticWeightedColumnViewsReady = false;
	sparseLogisticWeightedRows.clear();
	sparseLogisticWeightedColumnViews.clear();
	sparseLogisticWeightedRowsHaveUnitWeight = false;
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::ensureSparseLogisticWeightedColumnViews() {
	if (sparseLogisticWeightedColumnViewsReady) {
		return;
	}
	if (!std::is_same<BaseModel, LogisticRegression<RealType>>::value) {
		return;
	}
	ensureSparseLogisticColumnFormat();

	sparseLogisticWeightedRows.clear();
	sparseLogisticWeightedRows.reserve(K);
	sparseLogisticWeightedRowsHaveUnitWeight = true;
	for (size_t k = 0; k < K; ++k) {
		if (hKWeight[k] != static_cast<RealType>(0.0)) {
			sparseLogisticWeightedRows.push_back(static_cast<int>(k));
			if (hKWeight[k] != static_cast<RealType>(1.0)) {
				sparseLogisticWeightedRowsHaveUnitWeight = false;
			}
		}
	}
	sparseLogisticWeightedColumnViews.clear();
	sparseLogisticWeightedColumnViews.resize(J);
	for (int j = 0; j < static_cast<int>(J); ++j) {
		auto& view = sparseLogisticWeightedColumnViews[j];
		switch (sparseLogisticColumnFormat[j]) {
			case INDICATOR: {
				const int* rows = hX.getCompressedColumnVector(j);
				const size_t n = hX.getNumberOfEntries(j);
				view.rows.reserve(n);
				for (size_t row = 0; row < n; ++row) {
					const int k = rows[row];
					if (hKWeight[k] != static_cast<RealType>(0.0)) {
						view.rows.push_back(k);
					}
				}
				break;
			}
			case SPARSE: {
				const int* rows = hX.getCompressedColumnVector(j);
				const RealType* values = hX.getDataVector(j);
				const size_t n = hX.getNumberOfEntries(j);
				view.rows.reserve(n);
				view.values.reserve(n);
				for (size_t row = 0; row < n; ++row) {
					const int k = rows[row];
					if (hKWeight[k] != static_cast<RealType>(0.0)) {
						view.rows.push_back(k);
						view.values.push_back(values[row]);
					}
				}
				break;
			}
			case DENSE:
			case INTERCEPT:
				break;
		}
	}
	sparseLogisticWeightedColumnViewsReady = true;
}

template <class BaseModel,typename RealType>
RealType ModelSpecifics<BaseModel,RealType>::sparseLogisticLogOdds(RealType probability) const {
	const RealType one = static_cast<RealType>(1.0);
	const RealType zero = static_cast<RealType>(0.0);
	if (probability <= zero) {
		return -std::numeric_limits<RealType>::infinity();
	}
	if (probability >= one) {
		return std::numeric_limits<RealType>::infinity();
	}
	return std::log(probability) - std::log1p(-probability);
}

template <class BaseModel,typename RealType>
RealType ModelSpecifics<BaseModel,RealType>::sparseLogisticLogDenominator(RealType probability) const {
	const RealType one = static_cast<RealType>(1.0);
	const RealType zero = static_cast<RealType>(0.0);
	if (probability <= zero) {
		return zero;
	}
	if (probability >= one) {
		return std::numeric_limits<RealType>::infinity();
	}
	return -std::log1p(-probability);
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::refreshSparseLogisticState(bool useWeights) {
	if (!std::is_same<BaseModel, LogisticRegression<RealType>>::value) {
		return;
	}
	if (sparseLogisticStateReady) {
		return;
	}
	ensureSparseLogisticColumnFormat();
	if (sparseLogisticFitted.size() != K) {
		sparseLogisticFitted.resize(K);
	}
	if (sparseLogisticCaseRows.empty()) {
		sparseLogisticCaseRows.reserve(K);
		for (size_t i = 0; i < K; ++i) {
			if (hY[i] != static_cast<RealType>(0.0)) {
				sparseLogisticCaseRows.push_back(static_cast<int>(i));
			}
		}
	}
	for (size_t i = 0; i < K; ++i) {
		const RealType entry = offsExpXBeta[i];
		sparseLogisticFitted[i] = std::isfinite(entry)
			? entry / (static_cast<RealType>(1.0) + entry)
			: static_cast<RealType>(1.0);
	}
	sparseLogisticStateReady = true;
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::materializeSparseLogisticExpXBeta() {
	if (!sparseLogisticExpXBetaDirty) {
		return;
	}
	if (!std::is_same<BaseModel, LogisticRegression<RealType>>::value || !sparseLogisticStateReady) {
		sparseLogisticExpXBetaDirty = false;
		return;
	}

	const RealType one = static_cast<RealType>(1.0);
	const RealType zero = static_cast<RealType>(0.0);
	for (size_t i = 0; i < K; ++i) {
		const RealType p = sparseLogisticFitted[i];
		if (p <= zero) {
			offsExpXBeta[i] = zero;
		} else if (p >= one) {
			offsExpXBeta[i] = std::numeric_limits<RealType>::infinity();
		} else {
			offsExpXBeta[i] = p / (one - p);
		}
	}
	sparseLogisticExpXBetaDirty = false;
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::syncSparseLogisticState() {
	if (!sparseLogisticXBetaDirty) {
		return;
	}
	if (std::is_same<BaseModel, LogisticRegression<RealType>>::value) {
		materializeSparseLogisticExpXBeta();
		for (size_t i = 0; i < K; ++i) {
			hXBeta[i] = std::log(offsExpXBeta[i]);
			denomPid[i] = static_cast<RealType>(1.0) + offsExpXBeta[i];
		}
	}
	sparseLogisticXBetaDirty = false;
}

template <class BaseModel,typename RealType>
bool ModelSpecifics<BaseModel,RealType>::initializeXBetaFromRowSubset(
		const std::vector<double>& sourceXBeta,
		const std::vector<int>& keptRows) {
	if (keptRows.size() != K) {
		return false;
	}
	for (size_t i = 0; i < K; ++i) {
		const int oldRow = keptRows[i];
		if (oldRow < 0 || static_cast<size_t>(oldRow) >= sourceXBeta.size()) {
			return false;
		}
		hXBeta[i] = static_cast<RealType>(sourceXBeta[oldRow]);
	}
	sparseLogisticStateReady = false;
	sparseLogisticXBetaDirty = false;
	sparseLogisticExpXBetaDirty = false;
	return true;
}

template <class BaseModel,typename RealType>
bool ModelSpecifics<BaseModel,RealType>::initializeSparseLogisticStateFromRowSubset(
		const AbstractModelSpecifics& source,
		const std::vector<int>& keptRows) {
	if (!std::is_same<BaseModel, LogisticRegression<RealType>>::value
			|| keptRows.size() != K) {
		return false;
	}
	const auto* typedSource =
			dynamic_cast<const ModelSpecifics<BaseModel,RealType>*>(&source);
	if (typedSource == nullptr
			|| !typedSource->sparseLogisticStateReady
			|| typedSource->sparseLogisticFitted.empty()) {
		return false;
	}
	ensureSparseLogisticColumnFormat();
	if (sparseLogisticFitted.size() != K) {
		sparseLogisticFitted.resize(K);
	}
	for (size_t i = 0; i < K; ++i) {
		const int oldRow = keptRows[i];
		if (oldRow < 0
				|| static_cast<size_t>(oldRow) >= typedSource->sparseLogisticFitted.size()) {
			sparseLogisticStateReady = false;
			return false;
		}
		sparseLogisticFitted[i] = typedSource->sparseLogisticFitted[oldRow];
	}
	sparseLogisticCaseRows.clear();
	sparseLogisticCaseRows.reserve(K);
	for (size_t i = 0; i < K; ++i) {
		if (hY[i] != static_cast<RealType>(0.0)) {
			sparseLogisticCaseRows.push_back(static_cast<int>(i));
		}
	}
	sparseLogisticStateReady = true;
	sparseLogisticXBetaDirty = true;
	sparseLogisticExpXBetaDirty = true;
	return true;
}

template <class BaseModel,typename RealType>
bool ModelSpecifics<BaseModel,RealType>::hasSparseLogisticStateReady() const {
	return std::is_same<BaseModel, LogisticRegression<RealType>>::value
			&& sparseLogisticStateReady;
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeSparseLogisticGradientAndHessian(
		int index, double *ogradient, double *ohessian, bool useWeights) {
	if (!supportsSparseLogisticKernel(index, useWeights)) {
		computeGradientAndHessian(index, ogradient, ohessian, useWeights);
		return;
	}

	static const bool profileSparseLogisticHotLoops = []() {
		const char* env = std::getenv("CYCLOPS_SPARSE_LOGISTIC_PROFILE");
		return env != nullptr && env[0] != '\0' && env[0] != '0';
	}();
	std::chrono::steady_clock::time_point perfStart;
	if (profileSparseLogisticHotLoops) {
		perfStart = std::chrono::steady_clock::now();
	}
	refreshSparseLogisticState(useWeights);
	// Fold weights invalidate column views independently of fitted probabilities.
	if (useWeights) {
		ensureSparseLogisticWeightedColumnViews();
	}
	const RealType* fitted = sparseLogisticFitted.data();
	const RealType* weights = hKWeight.data();
	const RealType one = static_cast<RealType>(1.0);
	const bool computeHessian = ohessian != nullptr;

		RealType gradient = static_cast<RealType>(0.0);
		RealType hessian = static_cast<RealType>(0.0);

	switch (sparseLogisticColumnFormat[index]) {
		case INDICATOR: {
			const SparseLogisticWeightedColumnView* view =
				useWeights ? &sparseLogisticWeightedColumnViews[index] : nullptr;
			const int* rows = useWeights ? view->rows.data() : hX.getCompressedColumnVector(index);
				const size_t n = useWeights ? view->rows.size() : hX.getNumberOfEntries(index);
				if (!useWeights || sparseLogisticWeightedRowsHaveUnitWeight) {
					if (!sparseLogisticTryAvx2IndicatorGradientHessian(
							rows, fitted, n, computeHessian, gradient, hessian)) {
						RealType gradient0 = static_cast<RealType>(0.0);
						RealType gradient1 = static_cast<RealType>(0.0);
						RealType gradient2 = static_cast<RealType>(0.0);
						RealType gradient3 = static_cast<RealType>(0.0);
						RealType hessian0 = static_cast<RealType>(0.0);
						RealType hessian1 = static_cast<RealType>(0.0);
						RealType hessian2 = static_cast<RealType>(0.0);
						RealType hessian3 = static_cast<RealType>(0.0);
						size_t row = 0;
					for (; row + 3 < n; row += 4) {
						const int i0 = rows[row];
						const int i1 = rows[row + 1];
						const int i2 = rows[row + 2];
					const int i3 = rows[row + 3];
					const RealType p0 = fitted[i0];
					const RealType p1 = fitted[i1];
					const RealType p2 = fitted[i2];
					const RealType p3 = fitted[i3];
					gradient0 += p0;
					gradient1 += p1;
					gradient2 += p2;
					gradient3 += p3;
					if (computeHessian) {
						hessian0 += p0 * (one - p0);
						hessian1 += p1 * (one - p1);
						hessian2 += p2 * (one - p2);
						hessian3 += p3 * (one - p3);
					}
				}
				gradient += gradient0 + gradient1 + gradient2 + gradient3;
					if (computeHessian) {
						hessian += hessian0 + hessian1 + hessian2 + hessian3;
					}
					for (; row < n; ++row) {
						const int i = rows[row];
					const RealType p = fitted[i];
					gradient += p;
					if (computeHessian) {
							hessian += p * (one - p);
						}
					}
					}
				} else {
					RealType gradient0 = static_cast<RealType>(0.0);
				RealType gradient1 = static_cast<RealType>(0.0);
				RealType gradient2 = static_cast<RealType>(0.0);
				RealType gradient3 = static_cast<RealType>(0.0);
				RealType hessian0 = static_cast<RealType>(0.0);
				RealType hessian1 = static_cast<RealType>(0.0);
					RealType hessian2 = static_cast<RealType>(0.0);
					RealType hessian3 = static_cast<RealType>(0.0);
					size_t row = 0;
					for (; row + 3 < n; row += 4) {
						const int i0 = rows[row];
						const int i1 = rows[row + 1];
						const int i2 = rows[row + 2];
					const int i3 = rows[row + 3];
					const RealType p0 = fitted[i0];
					const RealType p1 = fitted[i1];
					const RealType p2 = fitted[i2];
					const RealType p3 = fitted[i3];
					const RealType w0 = weights[i0];
					const RealType w1 = weights[i1];
					const RealType w2 = weights[i2];
					const RealType w3 = weights[i3];
					gradient0 += w0 * p0;
					gradient1 += w1 * p1;
					gradient2 += w2 * p2;
					gradient3 += w3 * p3;
					if (computeHessian) {
						hessian0 += w0 * p0 * (one - p0);
						hessian1 += w1 * p1 * (one - p1);
						hessian2 += w2 * p2 * (one - p2);
						hessian3 += w3 * p3 * (one - p3);
					}
				}
				gradient += gradient0 + gradient1 + gradient2 + gradient3;
				if (computeHessian) {
					hessian += hessian0 + hessian1 + hessian2 + hessian3;
				}
				for (; row < n; ++row) {
					const int i = rows[row];
					const RealType p = fitted[i];
					const RealType w = weights[i];
					gradient += w * p;
					if (computeHessian) {
						hessian += w * p * (one - p);
					}
				}
			}
			break;
		}
		case SPARSE: {
			const SparseLogisticWeightedColumnView* view =
				useWeights ? &sparseLogisticWeightedColumnViews[index] : nullptr;
			const int* rows = useWeights ? view->rows.data() : hX.getCompressedColumnVector(index);
			const RealType* values = useWeights ? view->values.data() : hX.getDataVector(index);
			const size_t n = useWeights ? view->rows.size() : hX.getNumberOfEntries(index);
			for (size_t row = 0; row < n; ++row) {
				const int i = rows[row];
				const RealType x = values[row];
				const RealType p = fitted[i];
				const RealType w = (useWeights && !sparseLogisticWeightedRowsHaveUnitWeight) ? weights[i] : one;
				gradient += w * x * p;
				if (computeHessian) {
					hessian += w * x * x * p * (one - p);
				}
			}
			break;
		}
		case DENSE: {
			const RealType* values = hX.getDataVector(index);
			const size_t n = useWeights ? sparseLogisticWeightedRows.size() : K;
			for (size_t row = 0; row < n; ++row) {
				const int i = useWeights ? sparseLogisticWeightedRows[row] : static_cast<int>(row);
				const RealType x = values[i];
				const RealType p = fitted[i];
				const RealType w = (useWeights && !sparseLogisticWeightedRowsHaveUnitWeight) ? weights[i] : one;
				gradient += w * x * p;
				if (computeHessian) {
					hessian += w * x * x * p * (one - p);
				}
			}
			break;
		}
		case INTERCEPT: {
			const size_t n = useWeights ? sparseLogisticWeightedRows.size() : K;
			for (size_t row = 0; row < n; ++row) {
				const int i = useWeights ? sparseLogisticWeightedRows[row] : static_cast<int>(row);
				const RealType p = fitted[i];
				const RealType w = (useWeights && !sparseLogisticWeightedRowsHaveUnitWeight) ? weights[i] : one;
				gradient += w * p;
				if (computeHessian) {
					hessian += w * p * (one - p);
				}
			}
			break;
		}
	}

	gradient -= hXjY[index];
	*ogradient = static_cast<double>(gradient);
	if (ohessian != nullptr) {
		*ohessian = static_cast<double>(hessian);
	}
	if (profileSparseLogisticHotLoops) {
		const auto perfEnd = std::chrono::steady_clock::now();
		sparseLogisticPerformanceCounters.gradientHessianCalls += 1;
		sparseLogisticPerformanceCounters.nnzGradientHessian +=
			getSparseLogisticColumnScanSize(index, useWeights);
		sparseLogisticPerformanceCounters.gradientHessianSeconds +=
			std::chrono::duration<double>(perfEnd - perfStart).count();
	}
}

template <class BaseModel,typename RealType>
double ModelSpecifics<BaseModel,RealType>::updateSparseLogisticState(
		double delta, int index, bool useWeights) {
	if (!supportsSparseLogisticKernel(index, useWeights)) {
		updateXBeta(delta, index, useWeights);
		return 0.0;
	}

	static const bool profileSparseLogisticHotLoops = []() {
		const char* env = std::getenv("CYCLOPS_SPARSE_LOGISTIC_PROFILE");
		return env != nullptr && env[0] != '\0' && env[0] != '0';
	}();
	std::chrono::steady_clock::time_point perfStart;
	if (profileSparseLogisticHotLoops) {
		perfStart = std::chrono::steady_clock::now();
	}
	refreshSparseLogisticState(useWeights);
	static const int weightedUpdateRowsMode = []() {
		const char* env = std::getenv("CYCLOPS_WEIGHTED_UPDATE_ROWS");
		if (env == nullptr || env[0] == '\0') {
			return 1; // auto
		}
		if (env[0] == '0') {
			return 0; // disabled
		}
		return 2; // forced, for diagnostics
	}();
	bool useWeightedUpdateRows = false;
		if (useWeights && weightedUpdateRowsMode != 0) {
			ensureSparseLogisticWeightedColumnViews();
			useWeightedUpdateRows = weightedUpdateRowsMode == 2
				|| sparseLogisticWeightedRows.size() < K;
		}
		const RealType realDelta = static_cast<RealType>(delta);
			RealType* fitted = sparseLogisticFitted.data();
			const RealType one = static_cast<RealType>(1.0);
		const RealType zero = static_cast<RealType>(0.0);
		RealType fittedChange = static_cast<RealType>(0.0);

		auto rowWeight = [this, useWeights, useWeightedUpdateRows, one](const int row) {
			return (useWeights && useWeightedUpdateRows && !sparseLogisticWeightedRowsHaveUnitWeight)
				? hKWeight[row]
				: one;
		};

		auto advanceProbability = [one, zero](const RealType p, const RealType multiplier) {
			const RealType numerator = multiplier * p;
		const RealType denominator = (one - p) + numerator;
		return denominator > zero ? numerator / denominator : one;
		};

		auto updateProbability = [&fitted, &fittedChange, &rowWeight, &advanceProbability](
				const int row, const RealType multiplier) {
			const RealType oldP = fitted[row];
			const RealType newP = advanceProbability(oldP, multiplier);
			const RealType diff = newP - oldP;
			fitted[row] = newP;
			fittedChange += rowWeight(row) * diff * diff;
		};

		switch (sparseLogisticColumnFormat[index]) {
			case INDICATOR: {
				const RealType multiplier = std::exp(realDelta);
				const SparseLogisticWeightedColumnView* view =
					useWeightedUpdateRows ? &sparseLogisticWeightedColumnViews[index] : nullptr;
					const int* rows = useWeightedUpdateRows ? view->rows.data() : hX.getCompressedColumnVector(index);
					const size_t n = useWeightedUpdateRows ? view->rows.size() : hX.getNumberOfEntries(index);
				size_t row = 0;
				for (; row + 3 < n; row += 4) {
					const int i0 = rows[row];
					const int i1 = rows[row + 1];
					const int i2 = rows[row + 2];
					const int i3 = rows[row + 3];
					updateProbability(i0, multiplier);
					updateProbability(i1, multiplier);
					updateProbability(i2, multiplier);
					updateProbability(i3, multiplier);
				}
				for (; row < n; ++row) {
					const int i = rows[row];
					updateProbability(i, multiplier);
				}
			break;
		}
		case SPARSE: {
			const SparseLogisticWeightedColumnView* view =
				useWeightedUpdateRows ? &sparseLogisticWeightedColumnViews[index] : nullptr;
			const int* rows = useWeightedUpdateRows ? view->rows.data() : hX.getCompressedColumnVector(index);
			const RealType* values = useWeightedUpdateRows ? view->values.data() : hX.getDataVector(index);
			const size_t n = useWeightedUpdateRows ? view->rows.size() : hX.getNumberOfEntries(index);
			for (size_t row = 0; row < n; ++row) {
				const int i = rows[row];
				const RealType multiplier = std::exp(realDelta * values[row]);
				updateProbability(i, multiplier);
			}
			break;
		}
		case DENSE: {
			const RealType* values = hX.getDataVector(index);
			const size_t n = useWeightedUpdateRows ? sparseLogisticWeightedRows.size() : K;
			for (size_t row = 0; row < n; ++row) {
				const int i = useWeightedUpdateRows ? sparseLogisticWeightedRows[row] : static_cast<int>(row);
				const RealType multiplier = std::exp(realDelta * values[i]);
				updateProbability(i, multiplier);
			}
			break;
			}
			case INTERCEPT: {
				const RealType multiplier = std::exp(realDelta);
				const size_t n = useWeightedUpdateRows ? sparseLogisticWeightedRows.size() : K;
				for (size_t row = 0; row < n; ++row) {
					const int i = useWeightedUpdateRows ? sparseLogisticWeightedRows[row] : static_cast<int>(row);
				updateProbability(i, multiplier);
			}
			break;
		}
	}

	sparseLogisticExpXBetaDirty = true;
	sparseLogisticXBetaDirty = true;
		if (profileSparseLogisticHotLoops) {
			const auto perfEnd = std::chrono::steady_clock::now();
			sparseLogisticPerformanceCounters.updateCalls += 1;
			sparseLogisticPerformanceCounters.nnzUpdate +=
				getSparseLogisticColumnScanSize(index, useWeightedUpdateRows);
			sparseLogisticPerformanceCounters.updateSeconds +=
				std::chrono::duration<double>(perfEnd - perfStart).count();
		}
	return static_cast<double>(fittedChange);
}

// TODO The following function is an example of a double-dispatch, rewrite without need for virtual function
template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeGradientAndHessian(int index, double *ogradient,
		double *ohessian, bool useWeights) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

	if (hX.getNumberOfNonZeroEntries(index) == 0) {
	    *ogradient = 0.0; *ohessian = 0.0;
	    return;
	}
	ensureSparseIndices();

	// Run-time dispatch, so virtual call should not effect speed
	if (useWeights) {
		switch (hX.getFormatType(index)) {
			case INDICATOR :
				computeGradientAndHessianImpl<IndicatorIterator<RealType>>(index, ogradient, ohessian, weighted);
				break;
			case SPARSE :
				computeGradientAndHessianImpl<SparseIterator<RealType>>(index, ogradient, ohessian, weighted);
				break;
			case DENSE :
				computeGradientAndHessianImpl<DenseIterator<RealType>>(index, ogradient, ohessian, weighted);
				break;
			case INTERCEPT :
				computeGradientAndHessianImpl<InterceptIterator<RealType>>(index, ogradient, ohessian, weighted);
				break;
		}
	} else {
		switch (hX.getFormatType(index)) {
			case INDICATOR :
				computeGradientAndHessianImpl<IndicatorIterator<RealType>>(index, ogradient, ohessian, unweighted);
				break;
			case SPARSE :
				computeGradientAndHessianImpl<SparseIterator<RealType>>(index, ogradient, ohessian, unweighted);
				break;
			case DENSE :
				computeGradientAndHessianImpl<DenseIterator<RealType>>(index, ogradient, ohessian, unweighted);
				break;
			case INTERCEPT :
				computeGradientAndHessianImpl<InterceptIterator<RealType>>(index, ogradient, ohessian, unweighted);
				break;
		}
	}

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["compGradAndHess  "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
        duration["CPU GH           "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
}

template <class RealType>
std::pair<RealType, RealType> operator+(
        const std::pair<RealType, RealType>& lhs,
        const std::pair<RealType, RealType>& rhs) {

    return { lhs.first + rhs.first, lhs.second + rhs.second };
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeMMGradientAndHessian(
        std::vector<GradientHessian>& gh,
        const std::vector<bool>& fixBeta,
        bool useWeights) {

    if (norm.size() == 0) {
        initializeMM(boundType, fixBeta);
    }

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
    auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

    for (int index = 0; index < J; ++index) {
        double *ogradient = &(gh[index].first);
        double *ohessian  = &(gh[index].second);

        if (fixBeta[index]) {
            *ogradient = 0.0; *ohessian = 0.0;
        } else {

        // Run-time dispatch, so virtual call should not effect speed
        if (useWeights) {
            switch (hX.getFormatType(index)) {
            case INDICATOR :
                computeMMGradientAndHessianImpl<IndicatorIterator<RealType>>(index, ogradient, ohessian, weighted);
                break;
            case SPARSE :
                computeMMGradientAndHessianImpl<SparseIterator<RealType>>(index, ogradient, ohessian, weighted);
                break;
            case DENSE :
                computeMMGradientAndHessianImpl<DenseIterator<RealType>>(index, ogradient, ohessian, weighted);
                break;
            case INTERCEPT :
                computeMMGradientAndHessianImpl<InterceptIterator<RealType>>(index, ogradient, ohessian, weighted);
                break;
            }
        } else {
            switch (hX.getFormatType(index)) {
            case INDICATOR :
                computeMMGradientAndHessianImpl<IndicatorIterator<RealType>>(index, ogradient, ohessian, unweighted);
                break;
            case SPARSE :
                computeMMGradientAndHessianImpl<SparseIterator<RealType>>(index, ogradient, ohessian, unweighted);
                break;
            case DENSE :
                computeMMGradientAndHessianImpl<DenseIterator<RealType>>(index, ogradient, ohessian, unweighted);
                break;
            case INTERCEPT :
                computeMMGradientAndHessianImpl<InterceptIterator<RealType>>(index, ogradient, ohessian, unweighted);
                break;
            }
        }
        }
    }

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
    auto end = bsccs::chrono::steady_clock::now();
    ///////////////////////////"
    duration["compMMGradAndHess"] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

}

template <class BaseModel,typename RealType> template <class IteratorType, class Weights>
void ModelSpecifics<BaseModel,RealType>::computeMMGradientAndHessianImpl(int index, double *ogradient,
                                                                           double *ohessian, Weights w) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
    auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

    RealType gradient = static_cast<RealType>(0);
    RealType hessian = static_cast<RealType>(0);

    IteratorType it(hX, index);
    for (; it; ++it) {
        const int k = it.index();

        BaseModel::template incrementMMGradientAndHessian<IteratorType, Weights>(
                gradient, hessian, offsExpXBeta[k],
                denomPid[BaseModel::getGroup(hPid, k)], hNWeight[BaseModel::getGroup(hPid, k)],
                it.value(), hXBeta[k], hY[k], norm[k]);
    }

    if (BaseModel::precomputeGradient) { // Compile-time switch
        gradient -= hXjY[index];
    }

    if (BaseModel::precomputeHessian) { // Compile-time switch
        hessian += static_cast<RealType>(2.0) * hXjX[index];
    }

    *ogradient = static_cast<double>(gradient);
    *ohessian = static_cast<double>(hessian);

    // std::cerr << index << " " << gradient << " " << hessian << std::endl;

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
    auto end = bsccs::chrono::steady_clock::now();
    ///////////////////////////"
    auto name = "compGradHessMM" + IteratorType::name + "";
    duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif
}

template <class BaseModel,typename RealType> template <class IteratorType, class Weights>
void ModelSpecifics<BaseModel,RealType>::computeGradientAndHessianImpl(int index, double *ogradient,
		double *ohessian, Weights w) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

	RealType gradient = static_cast<RealType>(0);
	RealType hessian = static_cast<RealType>(0);

    if (BaseModel::cumulativeGradientAndHessian) { // Compile-time switch

        if (sparseIndices[index] == nullptr || sparseIndices[index]->size() > 0) {

    	    IteratorType it(sparseIndices[index].get(), N);

    	    RealType accNumerPid  = static_cast<RealType>(0);
            RealType accNumerPid2  = static_cast<RealType>(0);
            RealType decNumerPid = static_cast<RealType>(0); // ESK: Competing risks contrib to gradient
            RealType decNumerPid2 = static_cast<RealType>(0); // ESK: Competing risks contrib to hessian

            //computeBackwardAccumlatedNumerator(it, Weights::isWeighted); // Perform inside loop rather than outside.

    	    // find start relavent accumulator reset point
    	    auto reset = begin(accReset);
    	    while( *reset < it.index() ) {
    	        ++reset;
    	    }

    	    for (; it; ) {
    	        int i = it.index();

                if (*reset <= i) {
    	            accNumerPid  = static_cast<RealType>(0.0);
    	            accNumerPid2 = static_cast<RealType>(0.0);
    	            ++reset;
    	        }

    	        const auto numerator1 = numerPid[i];
                const auto numerator2 = numerPid2[i];

    	        accNumerPid += numerator1;
    	        accNumerPid2 += numerator2;

                // Compile-time delegation
                BaseModel::incrementGradientAndHessian(it,
                        w, // Signature-only, for iterator-type specialization
                        &gradient, &hessian, accNumerPid, accNumerPid2,
                        accDenomPid[i], hNWeight[i], 0.0, hXBeta[i], hY[i]); // When function is in-lined, compiler will only use necessary arguments

                // std::cerr << "oldA" << i << " " << accNumerPid << "/" << accDenomPid[i] << " " << hY[i] << " " << gradient << " " << hNWeight[i] << "\n";

                ++it;

    	        if (IteratorType::isSparse) {

    	            // TODO Should end be N or K?
    	            const int next = it ? it.index() : N;
                    for (++i; i < next; ++i) {

                        if (*reset <= i) {
    	                    accNumerPid  = static_cast<RealType>(0.0);
    	                    accNumerPid2 = static_cast<RealType>(0.0);
    	                    ++reset;
    	                }

    	                BaseModel::incrementGradientAndHessian(it,
                                w, // Signature-only, for iterator-type specialization
    	                        &gradient, &hessian, accNumerPid, accNumerPid2,
    	                        accDenomPid[i], hNWeight[i], static_cast<RealType>(0), hXBeta[i], hY[i]); // When function is in-lined, compiler will only use necessary arguments

                        // std::cerr << "oldB" << i << " " << accNumerPid << "/" << accDenomPid[i] << " " << hY[i] << " " << gradient << " " << hNWeight[i] << "\n";
    	            }
    	        }
    	    }
            if (BaseModel::isTwoWayScan) {
    	        // Manually perform backwards accumulation here instead of a separate function.
                auto revIt = it.reverse();

                // segmented prefix-scan
                RealType totalNumer = static_cast<RealType>(0);
                RealType totalNumer2 = static_cast<RealType>(0);

                auto backReset = end(accReset) - 1;

                for ( ; revIt; ) {

                    int i = revIt.index();

                    if (static_cast<signed int>(*backReset) == i) {
                        totalNumer = static_cast<RealType>(0);
                        totalNumer2 = static_cast<RealType>(0);
                        --backReset;
                    }

                    totalNumer += (hY[hNtoK[i]] > static_cast<RealType>(1)) ? numerPid[i] / hYWeight[hNtoK[i]] : 0;
                    totalNumer2 += (hY[hNtoK[i]] > static_cast<RealType>(1)) ? numerPid2[i] / hYWeight[hNtoK[i]]: 0;
                    decNumerPid = (hY[hNtoK[i]] == static_cast<RealType>(1)) ?
                                  hYWeight[hNtoK[i]] * totalNumer : 0;
                    decNumerPid2 = (hY[hNtoK[i]] == static_cast<RealType>(1)) ?
                                   hYWeight[hNtoK[i]] * totalNumer2 : 0;


                 // Increase gradient and hessian by competing risks contribution
                    BaseModel::incrementGradientAndHessian(it,
                                                           w, // Signature-only, for iterator-type specialization
                                                           &gradient, &hessian, decNumerPid, decNumerPid2,
                                                           accDenomPid[i], hNWeight[i], static_cast<RealType>(0), hXBeta[i],
                                                           hY[i]); // When function is in-lined, compiler will only use necess
                 --revIt;

                    if (IteratorType::isSparse) {

                        const int next = revIt ? revIt.index() : -1;
                        for (--i; i > next; --i) {
                            //if (*reset <= i) { // TODO MAS: This is not correct (only need for stratified models)
                            //    accNumerPid  = static_cast<RealType>(0);
                            //    accNumerPid2 = static_cast<RealType>(0);
                            //    ++reset;
                            //}

                           decNumerPid = (hY[hNtoK[i]] == static_cast<RealType>(1)) ? hYWeight[hNtoK[i]] * totalNumer : 0;
                           decNumerPid2 = (hY[hNtoK[i]] == static_cast<RealType>(1)) ? hYWeight[hNtoK[i]] * totalNumer2 : 0;
                           BaseModel::incrementGradientAndHessian(it, w, // Signature-only, for iterator-type specialization
                                                                   &gradient, &hessian, decNumerPid, decNumerPid2,
                                                                   accDenomPid[i], hNWeight[i], static_cast<RealType>(0),
                                                                   hXBeta[i],
                                                                   hY[i]); // When function is in-lined, compiler will only use necess
                        }
                    }
                } // End two-way scan
            } // End backwards iterator
		}

	} else if (BaseModel::hasIndependentRows) {

	    IteratorType it(hX, index);

	    for (; it; ++it) {
	        const int i = it.index();

	        RealType numerator1 = BaseModel::gradientNumeratorContrib(it.value(), offsExpXBeta[i], hXBeta[i], hY[i]);
	        RealType numerator2 = (!IteratorType::isIndicator && BaseModel::hasTwoNumeratorTerms) ?
	                BaseModel::gradientNumerator2Contrib(it.value(), offsExpXBeta[i]) : static_cast<RealType>(0);

	        // Compile-time delegation
	        BaseModel::incrementGradientAndHessian(it,
                    w, // Signature-only, for iterator-type specialization
                    &gradient, &hessian, numerator1, numerator2,
                    denomPid[i], hNWeight[i], it.value(), hXBeta[i], hY[i]); // When function is in-lined, compiler will only use necessary arguments
	    }

	} else if (BaseModel::exactCLR) {
	    //tbb::mutex mutex0;

#ifdef USE_RCPP_PARALLEL

	    tbb::combinable<RealType> newGrad(static_cast<RealType>(0));
	    tbb::combinable<RealType> newHess(static_cast<RealType>(0));

	    auto func = [&,index](const tbb::blocked_range<int>& range){

	        using std::isinf;

	        for (int i = range.begin(); i < range.end(); ++i) {
	            DenseView<IteratorType, RealType> x(IteratorType(hX, index), hNtoK[i], hNtoK[i+1]);
	            int numSubjects = hNtoK[i+1] - hNtoK[i];
	            int numCases = hNWeight[i];
	            std::vector<RealType> value = computeHowardRecursion<RealType>(offsExpXBeta.begin() + hNtoK[i], x, numSubjects, numCases);
	            if (value[0]==0 || value[1] == 0 || value[2] == 0 || isinf(value[0]) || isinf(value[1]) || isinf(value[2])) {
	                DenseView<IteratorType, RealType> newX(IteratorType(hX, index), hNtoK[i], hNtoK[i+1]);
	                std::vector<DDouble> value = computeHowardRecursion<DDouble>(offsExpXBeta.begin() + hNtoK[i], newX, numSubjects, numCases);//, threadPool);//, hY.begin() + hNtoK[i]);
	                using namespace sugar;
	                //mutex0.lock();
	                newGrad.local() -= (RealType)(-value[1]/value[0]);
	                newHess.local() -= (RealType)((value[1]/value[0]) * (value[1]/value[0]) - value[2]/value[0]);
	                //mutex0.unlock();
	                continue;
	            }
	            //mutex0.lock();
	            newGrad.local() -= (RealType)(-value[1]/value[0]);
	            newHess.local() -= (RealType)((value[1]/value[0]) * (value[1]/value[0]) - value[2]/value[0]);
	            //mutex0.unlock();
	        }
	    };
	    tbb::parallel_for(tbb::blocked_range<int>(0,N),func);
	    gradient += newGrad.combine([](const RealType& x, const RealType& y) {return x+y;});
	    hessian += newHess.combine([](const RealType& x, const RealType& y) {return x+y;});
#else

	    using std::isinf;

	    for (int i=0; i<N; i++) {
	    DenseView<IteratorType, RealType> x(IteratorType(hX, index), hNtoK[i], hNtoK[i+1]);
	    int numSubjects = hNtoK[i+1] - hNtoK[i];
	    int numCases = hNWeight[i];

	    std::vector<RealType> value = computeHowardRecursion<RealType>(offsExpXBeta.begin() + hNtoK[i], x, numSubjects, numCases);//, threadPool);//, hY.begin() + hNtoK[i]);
	    //std::cout<<" values" << i <<": "<<value[0]<<" | "<<value[1]<<" | "<<value[2];
	    if (value[0]==0 || value[1] == 0 || value[2] == 0 || isinf(value[0]) || isinf(value[1]) || isinf(value[2])) {
	    DenseView<IteratorType, RealType> newX(IteratorType(hX, index), hNtoK[i], hNtoK[i+1]);
	    std::vector<DDouble> value = computeHowardRecursion<DDouble>(offsExpXBeta.begin() + hNtoK[i], newX, numSubjects, numCases);//, threadPool);//, hY.begin() + hNtoK[i]);
	    using namespace sugar;
	    gradient -= (RealType)(-value[1]/value[0]);
	    hessian -= (RealType)((value[1]/value[0]) * (value[1]/value[0]) - value[2]/value[0]);
	    continue;
	    }
	    //gradient -= (RealType)(value[3] - value[1]/value[0]);
	    gradient -= (RealType)(-value[1]/value[0]);
	    hessian -= (RealType)((value[1]/value[0]) * (value[1]/value[0]) - value[2]/value[0]);
	    }
#endif // USE_RCPP_PARALLEL

	} else {

// 	    auto rangeKey = helper::dependent::getRangeKey(
// 	        hX, index, hPid,
//             typename IteratorType::tag());
//
// 	    auto rangeXNumerator = helper::dependent::getRangeX(
// 	        hX, index, offsExpXBeta,
//             typename IteratorType::tag());
//
// 	    auto rangeGradient = helper::dependent::getRangeGradient(
// 	        sparseIndices[index].get(), N, // runtime error: reference binding to null pointer of type 'struct vector'
//             denomPid, hNWeight,
//             typename IteratorType::tag());
//
// 	    const auto result = variants::trial::nested_reduce(
// 	        rangeKey.begin(), rangeKey.end(), // key
// 	        rangeXNumerator.begin(), // inner
// 	        rangeGradient.begin(), // outer
// 	        std::pair<RealType,RealType>{0,0}, Fraction<RealType>{0,0},
// 	        TestNumeratorKernel<BaseModel,IteratorType,RealType>(), // Inner transform-reduce
// 	        TestGradientKernel<BaseModel,IteratorType,Weights,RealType>()); // Outer transform-reduce
//
// 	    gradient = result.real();
// 	    hessian = result.imag();

	    IteratorType it(hX, index);
	    const int end = it.size() - 1;

	    RealType numerator1 = static_cast<RealType>(0);
	    RealType numerator2 = static_cast<RealType>(0);
	    int key = hPid[it.index()];

// 	    auto incrementNumerators2 = [this](const typename IteratorType::Index i, const typename IteratorType::ValueType x,
//                                            const RealTypePair lhs) -> RealTypePair {
//
// 	        const auto linearPredictor = offsExpXBeta[i];
// 	        return {
//     	       lhs.first + BaseModel::gradientNumeratorContrib(x, linearPredictor, static_cast<RealType>(0), static_cast<RealType>(0)),
// 	           lhs.second + (!IteratorType::isIndicator && BaseModel::hasTwoNumeratorTerms ?
// 	                BaseModel::gradientNumerator2Contrib(x, linearPredictor) :
// 	                static_cast<RealType>(0))
// 	        };
// 	    };

	    auto incrementNumerators = [this,&it,&numerator1,&numerator2]() {
	        const int i = it.index();

	        numerator1 += BaseModel::gradientNumeratorContrib(it.value(), offsExpXBeta[i], static_cast<RealType>(0), static_cast<RealType>(0));
	        numerator2 += (!IteratorType::isIndicator && BaseModel::hasTwoNumeratorTerms) ?
	        BaseModel::gradientNumerator2Contrib(it.value(), offsExpXBeta[i]) :
	            static_cast<RealType>(0);
	    };

	    for ( ; it.inRange(end); ++it) {
	        incrementNumerators();

	        const int nextKey = hPid[it.nextIndex()];
	        if (key != nextKey) {

	            BaseModel::incrementGradientAndHessian(it, w, // Signature-only, for iterator-type specialization
                                                    &gradient, &hessian, numerator1, numerator2,
                                                    denomPid[key], hNWeight[key], 0, 0, 0); // When function is in-lined, compiler will only use necessary arguments

	            numerator1 = static_cast<RealType>(0);
	            numerator2 = static_cast<RealType>(0);

	            key = nextKey;
	        }
	    }

	    incrementNumerators();

	    BaseModel::incrementGradientAndHessian(it, w, // Signature-only, for iterator-type specialization
                                            &gradient, &hessian, numerator1, numerator2,
                                            denomPid[key], hNWeight[key], 0, 0, 0); // When function is in-lined, compiler will only use necessary arguments
	}

	if (BaseModel::precomputeGradient) { // Compile-time switch
		gradient -= hXjY[index];
	}

	if (BaseModel::precomputeHessian) { // Compile-time switch
		hessian += static_cast<RealType>(2.0) * hXjX[index];
	}

 	*ogradient = static_cast<double>(gradient);
	*ohessian = static_cast<double>(hessian);

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	auto name = "compGradHess" + IteratorType::name + "  ";
	duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeThirdDerivative(int index, double *othird, bool useWeights) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
    auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

    if (hX.getNumberOfNonZeroEntries(index) == 0) {
        *othird = 0.0;
        return;
    }
    ensureSparseIndices();

    // Run-time dispatch, so virtual call should not effect speed
    if (useWeights) {
        switch (hX.getFormatType(index)) {
        case INDICATOR :
            computeThirdDerivativeImpl<IndicatorIterator<RealType>>(index, othird, weighted);
            break;
        case SPARSE :
            computeThirdDerivativeImpl<SparseIterator<RealType>>(index, othird, weighted);
            break;
        case DENSE :
            computeThirdDerivativeImpl<DenseIterator<RealType>>(index, othird, weighted);
            break;
        case INTERCEPT :
            computeThirdDerivativeImpl<InterceptIterator<RealType>>(index, othird, weighted);
            break;
        }
    } else {
        switch (hX.getFormatType(index)) {
        case INDICATOR :
            computeThirdDerivativeImpl<IndicatorIterator<RealType>>(index, othird, unweighted);
            break;
        case SPARSE :
            computeThirdDerivativeImpl<SparseIterator<RealType>>(index, othird, unweighted);
            break;
        case DENSE :
            computeThirdDerivativeImpl<DenseIterator<RealType>>(index, othird, unweighted);
            break;
        case INTERCEPT :
            computeThirdDerivativeImpl<InterceptIterator<RealType>>(index, othird, unweighted);
            break;
        }
    }

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
    auto end = bsccs::chrono::steady_clock::now();
    ///////////////////////////"
    duration["compThird  "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif
}

template <class BaseModel,typename RealType> template <class IteratorType, class Weights>
void ModelSpecifics<BaseModel,RealType>::computeThirdDerivativeImpl(int index, double *othird, Weights w) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
    auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

    RealType third = static_cast<RealType>(0);

    if (BaseModel::cumulativeGradientAndHessian) { // Compile-time switch

        if (sparseIndices[index] == nullptr || sparseIndices[index]->size() > 0) {

            IteratorType it(sparseIndices[index].get(), N);

            RealType accNumerPid  = static_cast<RealType>(0);
            RealType accNumerPid2  = static_cast<RealType>(0);

            // find start relavent accumulator reset point
            auto reset = begin(accReset);
            while( *reset < it.index() ) {
                ++reset;
            }

            for (; it; ) {
                int i = it.index();

                if (*reset <= i) {
                    accNumerPid  = static_cast<RealType>(0.0);
                    accNumerPid2 = static_cast<RealType>(0.0);
                    ++reset;
                }

                const auto numerator1 = numerPid[i];
                const auto numerator2 = numerPid2[i];

                accNumerPid += numerator1;
                accNumerPid2 += numerator2;

                // Compile-time delegation
                BaseModel::incrementThirdDerivative(it,
                                                    w, // Signature-only, for iterator-type specialization
                                                    &third, accNumerPid, accNumerPid2,
                                                    accDenomPid[i], hNWeight[i], 0.0, hXBeta[i], hY[i]); // When function is in-lined, compiler will only use necessary arguments
                ++it;

                if (IteratorType::isSparse) {

                    const int next = it ? it.index() : N;
                    for (++i; i < next; ++i) {

                        if (*reset <= i) {
                            accNumerPid  = static_cast<RealType>(0.0);
                            accNumerPid2 = static_cast<RealType>(0.0);
                            ++reset;
                        }

                        BaseModel::incrementThirdDerivative(it,
                                                            w, // Signature-only, for iterator-type specialization
                                                            &third, accNumerPid, accNumerPid2,
                                                            accDenomPid[i], hNWeight[i], static_cast<RealType>(0), hXBeta[i], hY[i]); // When function is in-lined, compiler will only use necessary arguments
                    }
                }
            }
        } else {
    		throw new std::logic_error("Not yet support");
        }
    }

    *othird = static_cast<double>(third);

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
    auto end = bsccs::chrono::steady_clock::now();
    ///////////////////////////"
    auto name = "compThird" + IteratorType::name + "  ";
    duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeSchoenfeldResiduals(int indexOne,
                                                                    std::vector<double>* residuals,
                                                                    std::vector<double>* otimes,
                                                                    std::vector<int>* strata,
                                                                    double* covariate,
                                                                    double* score,
                                                                    bool useWeights) {
    if (useWeights) {
        throw new std::logic_error("Weights are not yet implemented in Schoenfeld residual calculations");
    } else { // no weights
        switch (hX.getFormatType(indexOne)) {
        case INDICATOR :
            getSchoenfeldResidualsImpl<IndicatorIterator<RealType>>(indexOne, residuals, otimes, strata, covariate, score, weighted);
            break;
        case SPARSE :
            getSchoenfeldResidualsImpl<SparseIterator<RealType>>(indexOne, residuals, otimes, strata, covariate, score, weighted);
            break;
        case DENSE :
            getSchoenfeldResidualsImpl<DenseIterator<RealType>>(indexOne, residuals, otimes, strata, covariate, score, weighted);
            break;
        case INTERCEPT :
            getSchoenfeldResidualsImpl<InterceptIterator<RealType>>(indexOne, residuals, otimes, strata, covariate, score, weighted);
            break;
        }
    }
}


template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeFisherInformation(int indexOne, int indexTwo,
		double *oinfo, bool useWeights) {

	if (useWeights) {
		throw new std::logic_error("Weights are not yet implemented in Fisher Information calculations");
	} else { // no weights
		switch (hX.getFormatType(indexOne)) {
			case INDICATOR :
				dispatchFisherInformation<IndicatorIterator<RealType>>(indexOne, indexTwo, oinfo, weighted);
				break;
			case SPARSE :
				dispatchFisherInformation<SparseIterator<RealType>>(indexOne, indexTwo, oinfo, weighted);
				break;
			case DENSE :
				dispatchFisherInformation<DenseIterator<RealType>>(indexOne, indexTwo, oinfo, weighted);
				break;
			case INTERCEPT :
				dispatchFisherInformation<InterceptIterator<RealType>>(indexOne, indexTwo, oinfo, weighted);
				break;
		}
	}
}

template <class BaseModel, typename RealType> template <typename IteratorTypeOne, class Weights>
void ModelSpecifics<BaseModel,RealType>::dispatchFisherInformation(int indexOne, int indexTwo, double *oinfo, Weights w) {
	switch (hX.getFormatType(indexTwo)) {
		case INDICATOR :
			computeFisherInformationImpl<IteratorTypeOne,IndicatorIterator<RealType>>(indexOne, indexTwo, oinfo, w);
			break;
		case SPARSE :
			computeFisherInformationImpl<IteratorTypeOne,SparseIterator<RealType>>(indexOne, indexTwo, oinfo, w);
			break;
		case DENSE :
			computeFisherInformationImpl<IteratorTypeOne,DenseIterator<RealType>>(indexOne, indexTwo, oinfo, w);
			break;
		case INTERCEPT :
			computeFisherInformationImpl<IteratorTypeOne,InterceptIterator<RealType>>(indexOne, indexTwo, oinfo, w);
			break;
	}
//	std::cerr << "End of dispatch" << std::endl;
}


template<class BaseModel, typename RealType> template<class IteratorType>
SparseIterator<RealType> ModelSpecifics<BaseModel, RealType>::getSubjectSpecificHessianIterator(int index) {

	if (hessianSparseCrossTerms.find(index) == hessianSparseCrossTerms.end()) {

        auto indices = make_shared<std::vector<int> >();
        auto values = make_shared<std::vector<RealType> >();

    	CDCPtr column = bsccs::make_shared<CompressedDataColumn<RealType>>(indices, values, SPARSE);
		hessianSparseCrossTerms.insert(std::make_pair(index, column));

		IteratorType itCross(hX, index);
		for (; itCross;) {
			RealType value = 0.0;
			int currentPid = hPid[itCross.index()];  // TODO Need to fix for stratified Cox
			do {
				const int k = itCross.index();
				value += BaseModel::gradientNumeratorContrib(itCross.value(),
						offsExpXBeta[k], hXBeta[k], hY[k]);
				++itCross;
			} while (itCross && currentPid == hPid[itCross.index()]); // TODO Need to fix for stratified Cox
			indices->push_back(currentPid);
			values->push_back(value);
		}
	}
	return SparseIterator<RealType>(*hessianSparseCrossTerms[index]);

}

template <class BaseModel, typename RealType> template <class IteratorTypeOne, class IteratorTypeTwo, class Weights>
void ModelSpecifics<BaseModel,RealType>::computeFisherInformationImpl(int indexOne, int indexTwo, double *oinfo, Weights w) {

	IteratorTypeOne itOne(hX, indexOne);
	IteratorTypeTwo itTwo(hX, indexTwo);
	PairProductIterator<IteratorTypeOne,IteratorTypeTwo,RealType> it(itOne, itTwo);

	RealType information = static_cast<RealType>(0);
	for (; it.valid(); ++it) {
		const int k = it.index();
		// Compile-time delegation

		BaseModel::incrementFisherInformation(it,
				w, // Signature-only, for iterator-type specialization
				&information,
				offsExpXBeta[k],
				0.0, 0.0, // numerPid[k], numerPid2[k], // remove
				denomPid[BaseModel::getGroup(hPid, k)],
				hKWeight[k], it.value(), hXBeta[k], hY[k]); // When function is in-lined, compiler will only use necessary arguments
	}

	if (BaseModel::hasStrataCrossTerms) {

		// Check if index is pre-computed
//#define USE_DENSE
#ifdef USE_DENSE
		if (hessianCrossTerms.find(indexOne) == hessianCrossTerms.end()) {
			// Make new
			std::vector<real> crossOneTerms(N);
			IteratorTypeOne crossOne(modelData, indexOne);
			for (; crossOne; ++crossOne) {
				const int k = crossOne.index();
				incrementByGroup(crossOneTerms.data(), hPid, k,
						BaseModel::gradientNumeratorContrib(crossOne.value(), offsExpXBeta[k], hXBeta[k], hY[k]));
			}
			hessianCrossTerms[indexOne];
			hessianCrossTerms[indexOne].swap(crossOneTerms);
		}
		std::vector<real>& crossOneTerms = hessianCrossTerms[indexOne];

		// TODO Remove code duplication
		if (hessianCrossTerms.find(indexTwo) == hessianCrossTerms.end()) {
			std::vector<real> crossTwoTerms(N);
			IteratorTypeTwo crossTwo(modelData, indexTwo);
			for (; crossTwo; ++crossTwo) {
				const int k = crossTwo.index();
				incrementByGroup(crossTwoTerms.data(), hPid, k,
						BaseModel::gradientNumeratorContrib(crossTwo.value(), offsExpXBeta[k], hXBeta[k], hY[k]));
			}
			hessianCrossTerms[indexTwo];
			hessianCrossTerms[indexTwo].swap(crossTwoTerms);
		}
		std::vector<real>& crossTwoTerms = hessianCrossTerms[indexTwo];

		// TODO Sparse loop
		real cross = 0.0;
		for (int n = 0; n < N; ++n) {
			cross += crossOneTerms[n] * crossTwoTerms[n] / (denomPid[n] * denomPid[n]);
		}
		information -= cross;
#else
		SparseIterator<RealType> sparseCrossOneTerms = getSubjectSpecificHessianIterator<IteratorTypeOne>(indexOne);
		SparseIterator<RealType> sparseCrossTwoTerms = getSubjectSpecificHessianIterator<IteratorTypeTwo>(indexTwo);
		PairProductIterator<SparseIterator<RealType>,SparseIterator<RealType>,RealType> itSparseCross(sparseCrossOneTerms, sparseCrossTwoTerms);

		RealType sparseCross = static_cast<RealType>(0);
		for (; itSparseCross.valid(); ++itSparseCross) {
			const int n = itSparseCross.index();
			sparseCross += itSparseCross.value() / (denomPid[n] * denomPid[n]);
		}
		information -= sparseCross;
#endif
	}

	*oinfo = static_cast<double>(information);
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeNumeratorForGradient(int index, bool useWeights) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

	ensureSparseIndices();
	if (BaseModel::hasNtoKIndices || BaseModel::cumulativeGradientAndHessian) {

		switch (hX.getFormatType(index)) {
		case INDICATOR : {
				IndicatorIterator<RealType> itI(*(sparseIndices)[index]);
				for (; itI; ++itI) { // Only affected entries
					numerPid[itI.index()] = static_cast<RealType>(0.0);
				}
/*
				zeroVector(numerPid.data(), N);
				if (BaseModel::efron) {
					zeroVector(numerPid3.data(), N);
				}
*/
				if (useWeights) {
				    incrementNumeratorForGradientImpl<IndicatorIterator<RealType>, WeightedOperation>(index);
				} else {
				    incrementNumeratorForGradientImpl<IndicatorIterator<RealType>, UnweightedOperation>(index);
				}
				break;
		}
		case SPARSE : {
				SparseIterator<RealType> itS(*(sparseIndices)[index]);
				for (; itS; ++itS) { // Only affected entries
					numerPid[itS.index()] = static_cast<RealType>(0.0);
					if (BaseModel::hasTwoNumeratorTerms) { // Compile-time switch
						numerPid2[itS.index()] = static_cast<RealType>(0.0); // TODO Does this invalid the cache line too much?
					}
				}
/*
				zeroVector(numerPid.data(), N);
				if (BaseModel::hasTwoNumeratorTerms) { // Compile-time switch
					zeroVector(numerPid2.data(), N);
				}
				if (BaseModel::efron) {
					zeroVector(numerPid3.data(), N);
					zeroVector(numerPid4.data(), N);
				}
*/
				if (useWeights) {
				    incrementNumeratorForGradientImpl<SparseIterator<RealType>, WeightedOperation>(index);
				} else {
				    incrementNumeratorForGradientImpl<SparseIterator<RealType>, UnweightedOperation>(index);
				}
				break;
		}
		case DENSE : {
				zeroVector(numerPid.data(), N);
				if (BaseModel::hasTwoNumeratorTerms) { // Compile-time switch
					zeroVector(numerPid2.data(), N);
				}
/*
				if (BaseModel::efron) {
					zeroVector(numerPid3.data(), N);
					zeroVector(numerPid4.data(), N);
				}
*/
				if (useWeights) {
				    incrementNumeratorForGradientImpl<DenseIterator<RealType>, WeightedOperation>(index);
				} else {
				    incrementNumeratorForGradientImpl<DenseIterator<RealType>, UnweightedOperation>(index);
				}
				break;
		}
		case INTERCEPT : {
				zeroVector(numerPid.data(), N);
				if (BaseModel::hasTwoNumeratorTerms) { // Compile-time switch
					zeroVector(numerPid2.data(), N);
				}
/*
				if (BaseModel::efron) {
					zeroVector(numerPid3.data(), N);
					zeroVector(numerPid4.data(), N);
				}
*/
				if (useWeights) {
				    incrementNumeratorForGradientImpl<InterceptIterator<RealType>, WeightedOperation>(index);
				} else {
				    incrementNumeratorForGradientImpl<InterceptIterator<RealType>, UnweightedOperation>(index);
				}
				break;
		}
		default : break;
		}
	}

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["compNumForGrad   "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif
#endif

#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	duration["CPU GH           "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
}

template <class BaseModel,typename RealType> template <class IteratorType, class Weights>
void ModelSpecifics<BaseModel,RealType>::incrementNumeratorForGradientImpl(int index) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

	IteratorType it(hX, index);
	for (; it; ++it) {
		const int k = it.index();
		incrementByGroup(numerPid.data(), hPid, k,
                   Weights::isWeighted ?
                       hKWeight[k] * BaseModel::gradientNumeratorContrib(it.value(), offsExpXBeta[k], hXBeta[k], hY[k]) :
		               BaseModel::gradientNumeratorContrib(it.value(), offsExpXBeta[k], hXBeta[k], hY[k])
		    );
		if (!IteratorType::isIndicator && BaseModel::hasTwoNumeratorTerms) {
			incrementByGroup(numerPid2.data(), hPid, k,
                    Weights::isWeighted ?
                        hKWeight[k] * BaseModel::gradientNumerator2Contrib(it.value(), offsExpXBeta[k]) :
                        BaseModel::gradientNumerator2Contrib(it.value(), offsExpXBeta[k])
                );
		}
/*
		if (BaseModel::efron) {
			incrementByGroup(numerPid3.data(), hPid, k,
					Weights::isWeighted ?
						hKWeight[k] * hY[k] * BaseModel::gradientNumeratorContrib(it.value(), offsExpXBeta[k], hXBeta[k], hY[k]) :
						hY[k] * BaseModel::gradientNumeratorContrib(it.value(), offsExpXBeta[k], hXBeta[k], hY[k])
				);
		}
		if (!IteratorType::isIndicator && BaseModel::efron) {
			incrementByGroup(numerPid4.data(), hPid, k,
                    Weights::isWeighted ?
                        hKWeight[k] * hY[k] * BaseModel::gradientNumerator2Contrib(it.value(), offsExpXBeta[k]) :
                        hY[k] * BaseModel::gradientNumerator2Contrib(it.value(), offsExpXBeta[k])
                );
		}
*/
	}
#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	auto name = "compNumGrad" + IteratorType::name + "   ";
	duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::updateXBeta(double delta, int index, bool useWeights) {
	syncSparseLogisticState();
	sparseLogisticStateReady = false;

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

	RealType realDelta = static_cast<RealType>(delta);

	// Run-time dispatch to implementation depending on covariate FormatType
	switch(hX.getFormatType(index)) {
	    case INDICATOR : {
		    if (useWeights) {
			    updateXBetaImpl<IndicatorIterator<RealType>, WeightedOperation>(realDelta, index);
		    } else {
		        updateXBetaImpl<IndicatorIterator<RealType>, UnweightedOperation>(realDelta, index);
		    }
			break;
	    }
	    case SPARSE : {
		    if (useWeights) {
			    updateXBetaImpl<SparseIterator<RealType>, WeightedOperation>(realDelta, index);
		    } else {
		        updateXBetaImpl<SparseIterator<RealType>, UnweightedOperation>(realDelta, index);
		    }
			break;
	    }
	    case DENSE : {
	        if (useWeights) {
			    updateXBetaImpl<DenseIterator<RealType>, WeightedOperation>(realDelta, index);
	        } else {
	            updateXBetaImpl<DenseIterator<RealType>, UnweightedOperation>(realDelta, index);
	        }
			break;
	    }
	    case INTERCEPT : {
	        if (useWeights) {
			    updateXBetaImpl<InterceptIterator<RealType>, WeightedOperation>(realDelta, index);
	        } else {
	            updateXBetaImpl<InterceptIterator<RealType>, UnweightedOperation>(realDelta, index);
	        }
			break;
	    }
		default : break;
	}

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["updateXBeta      "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	duration["CPU GH           "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

}

template <class BaseModel,typename RealType> template <class IteratorType, class Weights>
inline void ModelSpecifics<BaseModel,RealType>::updateXBetaImpl(RealType realDelta, int index) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

	IteratorType it(hX, index);

    if (BaseModel::cumulativeGradientAndHessian) { // cox
        for (; it; ++it) {
            const int k = it.index();
            hXBeta[k] += realDelta * it.value(); // TODO Check optimization with indicator and intercept

            if (BaseModel::likelihoodHasDenominator) { // Compile-time switch
                RealType oldEntry = Weights::isWeighted ?
                    hKWeight[k] * offsExpXBeta[k] : offsExpXBeta[k]; // TODO Delegate condition to forming offExpXBeta
                offsExpXBeta[k] = BaseModel::getOffsExpXBeta(hOffs.data(), hXBeta[k], hY[k], k); // Update offsExpXBeta
                RealType newEntry = Weights::isWeighted ?
                    hKWeight[k] * offsExpXBeta[k] : offsExpXBeta[k]; // TODO Delegate condition
                incrementByGroup(denomPid.data(), hPid, k, (newEntry - oldEntry)); // Update denominators
            }
        }
    } else {
        for (; it; ++it) {
            const int k = it.index();
            hXBeta[k] += realDelta * it.value(); // TODO Check optimization with indicator and intercept

            if (BaseModel::likelihoodHasDenominator) { // Compile-time switch
                RealType oldEntry = offsExpXBeta[k];
                RealType newEntry = offsExpXBeta[k] = BaseModel::getOffsExpXBeta(hOffs.data(), hXBeta[k], hY[k], k);
                incrementByGroup(denomPid.data(), hPid, k, (newEntry - oldEntry));
/*
                if (BaseModel::efron) {
                    incrementByGroup(denomPid2.data(), hPid, k, hY[k]*(newEntry - oldEntry)); // Update denominators
                }
*/
            }
        }
    }

	computeAccumlatedDenominator(Weights::isWeighted);

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	auto name = "updateXBeta" + IteratorType::name + "   ";
	duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

}

template <class BaseModel, typename RealType> template <class Weights>
void ModelSpecifics<BaseModel,RealType>::computeRemainingStatisticsImpl() {

    auto& xBeta = getXBeta();

    if (BaseModel::likelihoodHasDenominator) {
        fillVector(denomPid.data(), N, BaseModel::getDenomNullValue());
/*
        if (BaseModel::efron) {
        	fillVector(denomPid2.data(), N, BaseModel::getDenomNullValue());
        }
*/
        if (BaseModel::cumulativeGradientAndHessian) { // cox
            for (size_t k = 0; k < K; ++k) {
                offsExpXBeta[k] = BaseModel::getOffsExpXBeta(hOffs.data(), xBeta[k], hY[k], k);
                RealType weightoffsExpXBeta =  Weights::isWeighted ?
                    hKWeight[k] * BaseModel::getOffsExpXBeta(hOffs.data(), xBeta[k], hY[k], k) :
                    BaseModel::getOffsExpXBeta(hOffs.data(), xBeta[k], hY[k], k);
                incrementByGroup(denomPid.data(), hPid, k, weightoffsExpXBeta); // Update denominators
            }
        } else {
            for (size_t k = 0; k < K; ++k) {
                offsExpXBeta[k] = BaseModel::getOffsExpXBeta(hOffs.data(), xBeta[k], hY[k], k);
                incrementByGroup(denomPid.data(), hPid, k, offsExpXBeta[k]);
/*
                if (BaseModel::efron) {
                    incrementByGroup(denomPid2.data(), hPid, k, hY[k]*offsExpXBeta[k]); // Update denominators
                }
*/
            }
        }
        computeAccumlatedDenominator(Weights::isWeighted);
    }
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeRemainingStatistics(bool useWeights) {

#ifdef CYCLOPS_DEBUG_TIMING
	auto start = bsccs::chrono::steady_clock::now();
#endif

	if (useWeights) {
	    computeRemainingStatisticsImpl<WeightedOperation>();
	} else {
	    computeRemainingStatisticsImpl<UnweightedOperation>();
	}

#ifdef CYCLOPS_DEBUG_TIMING
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["compRS           "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif

}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeAccumlatedNumerator(bool useWeights) {

	if (BaseModel::likelihoodHasDenominator && //The two switches should ideally be separated
			BaseModel::cumulativeGradientAndHessian) { // Compile-time switch

		if (accNumerPid.size() != N) {
			accNumerPid.resize(N, static_cast<RealType>(0));
		}
		if (accNumerPid2.size() != N) {
			accNumerPid2.resize(N, static_cast<RealType>(0));
		}

		// segmented prefix-scan
		RealType totalNumer = static_cast<RealType>(0);
		RealType totalNumer2 = static_cast<RealType>(0);

		auto reset = begin(accReset);

		for (size_t i = 0; i < N; ++i) {

			if (static_cast<unsigned int>(*reset) == i) {
				totalNumer = static_cast<RealType>(0);
				totalNumer2 = static_cast<RealType>(0);
				++reset;
			}

			totalNumer += numerPid[i];
			totalNumer2 += numerPid2[i];
			accNumerPid[i] = totalNumer;
			accNumerPid2[i] = totalNumer2;
        }
	}
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeAccumlatedDenominator(bool useWeights) {

    if (BaseModel::likelihoodHasDenominator && //The two switches should ideally be separated
        BaseModel::cumulativeGradientAndHessian) { // Compile-time switch


        if (accDenomPid.size() != (N + 1)) {
            accDenomPid.resize(N + 1, static_cast<RealType>(0));
        }

        // segmented prefix-scan
        RealType totalDenom = static_cast<RealType>(0);

        auto reset = begin(accReset);

        for (size_t i = 0; i < N; ++i) {

            if (static_cast<unsigned int>(*reset) == i) { // TODO Check with SPARSE
                totalDenom = static_cast<RealType>(0);
                ++reset;
            }

            totalDenom += denomPid[i];
            accDenomPid[i] = totalDenom;
        }
        //ESK : Incorporate backwards scan here:
        if (BaseModel::isTwoWayScan) {
            // RealType backDenom = static_cast<RealType>(0);
            RealType totalDenom = static_cast<RealType>(0);

        auto reset = begin(accReset);

            //Q: How can we change int to size_t w/o errors
            for (int i = (N - 1); i >= 0; i--) {

                if (static_cast<unsigned int>(*reset) == i) { // TODO Check with SPARSE
                    totalDenom = static_cast<RealType>(0);
                    ++reset;
                }

                totalDenom += (hY[hNtoK[i]] > static_cast<RealType>(1)) ? denomPid[i] / hYWeight[hNtoK[i]] : 0;
                accDenomPid[i] += (hY[hNtoK[i]] == static_cast<RealType>(1)) ? hYWeight[hNtoK[i]] * totalDenom : 0;
            }
        } // End two-way scan
    }

}

// ESK: This is no longer needed. Incorporated in incrementGradientAndHessian
template <class BaseModel,typename RealType> template <class IteratorType>
void ModelSpecifics<BaseModel,RealType>::computeBackwardAccumlatedNumerator(
        IteratorType it,
        bool useWeights) {

    if (decNumerPid.size() != N) {
        decNumerPid.resize(N, static_cast<RealType>(0));
    }
    if (decNumerPid2.size() != N) {
        decNumerPid2.resize(N, static_cast<RealType>(0));
    }
    auto revIt = it.reverse();

    // segmented prefix-scan
    RealType totalNumer = static_cast<RealType>(0);
    RealType totalNumer2 = static_cast<RealType>(0);

    auto reset = end(accReset) - 1;

    for ( ; revIt; ) {

        int i = revIt.index();

        if (static_cast<signed int>(*reset) == i) {
            totalNumer = static_cast<RealType>(0);
            totalNumer2 = static_cast<RealType>(0);
            --reset;
        }

        totalNumer +=  (hY[hNtoK[i]] > static_cast<RealType>(1)) ? numerPid[i] / hYWeight[hNtoK[i]] : 0;
        totalNumer2 += (hY[hNtoK[i]] > static_cast<RealType>(1)) ? numerPid2[i] / hYWeight[hNtoK[i]] : 0;
        decNumerPid[i] = (hY[hNtoK[i]] == static_cast<RealType>(1)) ? hYWeight[hNtoK[i]] * totalNumer : 0;
        decNumerPid2[i] = (hY[hNtoK[i]] == static_cast<RealType>(1)) ? hYWeight[hNtoK[i]] * totalNumer2 : 0;
        --revIt;

        if (IteratorType::isSparse) {

            const int next = revIt ? revIt.index() : -1;
            for (--i; i > next; --i) { // TODO MAS: This may be incorrect
                //if (*reset <= i) { // TODO MAS: This is not correct (only need for stratifed models)
                //    accNumerPid  = static_cast<RealType>(0);
                //    accNumerPid2 = static_cast<RealType>(0);
                //    ++reset;
                //}

                // ESK: Start implementing sparse
                decNumerPid[i] = (hY[hNtoK[i]] == static_cast<RealType>(1)) ? hYWeight[hNtoK[i]] * totalNumer : 0;
                decNumerPid2[i] = (hY[hNtoK[i]] == static_cast<RealType>(1)) ? hYWeight[hNtoK[i]] * totalNumer2 : 0;
            }
        }
    }
}

// ESK: This is no longer needed. Incorporated in computeAccumlatedDenominator
template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::computeBackwardAccumlatedDenominator(bool useWeights) {

    if (BaseModel::likelihoodHasDenominator && //The two switches should ideally be separated
            BaseModel::cumulativeGradientAndHessian) { // Compile-time switch

        // segmented prefix-scan
        RealType backDenom = static_cast<RealType>(0);
        RealType totalDenom = static_cast<RealType>(0);

        auto reset = begin(accReset);

        //Q: How can we change int to size_t w/o errors
        for (int i = (N - 1); i >= 0; i--) {

            if (static_cast<unsigned int>(*reset) == i) { // TODO Check with SPARSE
                totalDenom = static_cast<RealType>(0);
                ++reset;
            }

            totalDenom += (hY[hNtoK[i]] > static_cast<RealType>(1)) ? denomPid[i] / hYWeight[hNtoK[i]] : 0;
            accDenomPid[i] += (hY[hNtoK[i]] == static_cast<RealType>(1)) ? hYWeight[hNtoK[i]] * totalDenom : 0;
        }
    }
}


template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::doSortPid(bool useCrossValidation) {
/* For Cox model:
 *
 * We currently assume that hZ[k] are sorted in decreasing order by k.
 *
 */
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::getOriginalPid() {

    hPidInternal =  hPidOriginal; // Make copy

}
template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::setPidForAccumulation(const double* weights) {
	setPidForAccumulationImpl(weights);
}

template <class BaseModel,typename RealType> template <typename AnyRealType>
void ModelSpecifics<BaseModel,RealType>::setPidForAccumulationImpl(const AnyRealType* weights) {

    hPidInternal =  hPidOriginal; // Make copy
    hPid = hPidInternal.data(); // Point to copy
    hPidSize = hPidInternal.size();
    accReset.clear();

    const int ignore = -1;

    // Find first non-zero weight
    size_t index = 0;

    while(weights != nullptr && weights[index] == 0.0 && index < K) {
        hPid[index] = ignore;
        index++;
    }

    int lastPid = hPid[index];
    AnyRealType lastTime = hOffs[index];
    AnyRealType lastEvent = hY[index];

    int pid = hPid[index] = 0;

    int nTies = 0;

    for (size_t k = index + 1; k < K; ++k) {
        if (weights == nullptr || weights[k] != 0.0) {
            int nextPid = hPid[k];

            if (nextPid != lastPid) { // start new strata
                pid++;
                accReset.push_back(pid);
                nTies = 0;
                lastPid = nextPid;
            } else {

                if (lastEvent == 1.0 && lastTime == hOffs[k] && lastEvent == hY[k]) {
                    // In a tie, do not increment denominator
                    ++nTies;
                } else {
                    pid++;
                }
            }
            lastTime = hOffs[k];
            lastEvent = hY[k];

            hPid[k] = pid;
        } else {
            hPid[k] = ignore;
        }
    }
    pid++;
    accReset.push_back(pid);

    // Save number of denominators
    N = pid;

    if (weights != nullptr) {
        for (size_t i = 0; i < K; ++i) {
            if (hPid[i] == ignore) hPid[i] = N; // do NOT accumulate, since loops use: i < N
        }
    }
    setupSparseIndices(N); // ignore pid == N (pointing to removed data strata)
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::setupSparseIndices(const int max) {
    sparseIndices.clear(); // empty if full!

    for (size_t j = 0; j < J; ++j) {
        if (hX.getFormatType(j) == DENSE || hX.getFormatType(j) == INTERCEPT) {
            sparseIndices.push_back(NULL);
        } else {
            std::set<int> unique;
            const size_t n = hX.getNumberOfEntries(j);
            const int* indicators = hX.getCompressedColumnVector(j);
            for (size_t j = 0; j < n; j++) { // Loop through non-zero entries only
                const int k = indicators[j];
                const int i = (k < hPidSize) ? hPid[k] : k;
                if (i < max) {
                    unique.insert(i);
                }
            }
            auto indices = bsccs::make_shared<IndexVector>(unique.begin(), unique.end());
            sparseIndices.push_back(indices);
        }
    }
    sparseIndicesReady = true;
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::ensureSparseIndices() {
    if (!sparseIndicesReady) {
        setupSparseIndices(N);
    }
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::deviceInitialization() {
    // Do nothing
}

template <class BaseModel,typename RealType>
void ModelSpecifics<BaseModel,RealType>::initialize(
        int iN,
        int iK,
        int iJ,
        const void*,
        double* iNumerPid,
        double* iNumerPid2,
        double* iDenomPid,
        double* iXjY,
        std::vector<std::vector<int>* >* iSparseIndices,
        const int* iPid_unused,
        double* iOffsExpXBeta,
        double* iXBeta,
        double* iOffs,
        double* iBeta,
        const double* iY_unused) {

    N = iN;
    K = iK;
    J = iJ;
    offsExpXBeta.resize(K);
    hXBeta.resize(K);

    if (allocateXjY()) {
        hXjY.resize(J);
    }

    if (allocateXjX()) {
        hXjX.resize(J);
    }
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto start = std::chrono::steady_clock::now();
#endif
    if (initializeAccumulationVectors()) {
        setPidForAccumulation(static_cast<double*>(nullptr)); // calls setupSparseIndices() before returning
    } else {
        // TODO Suspect below is not necessary for non-grouped data.
        // If true, then fill with pointers to CompressedDataColumn and do not delete in destructor
        if (std::is_same<BaseModel, LogisticRegression<RealType>>::value) {
            sparseIndices.clear();
            sparseIndicesReady = false;
        } else {
            setupSparseIndices(N); // Need to be recomputed when hPid change!
        }
    }
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto end = std::chrono::steady_clock::now();
	double timerPid = std::chrono::duration<double>(end - start).count();
	std::cout << " OVERHEAD CCD setPid:  " << timerPid << " s \n";
#endif


    size_t alignedLength = getAlignedLength(N + 1);
    // 	numerDenomPidCache.resize(3 * alignedLength, 0);
    // 	numerPid = numerDenomPidCache.data();
    // 	denomPid = numerPid + alignedLength; // Nested in denomPid allocation
    // 	numerPid2 = numerPid + 2 * alignedLength;
    denomPid.resize(alignedLength);
    denomPid2.resize(alignedLength);
    numerPid.resize(alignedLength);
    numerPid2.resize(alignedLength);
    numerPid3.resize(alignedLength);
    numerPid4.resize(alignedLength);

    deviceInitialization();

}

} // namespace

#endif /* MODELSPECIFICS_HPP_ */
