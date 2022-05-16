#pragma once
#ifndef __CUDACC__
#error __FILE__ can only be compiled by nvcc and nvrtc exclusively
#endif

#ifndef _STP_SINGLE_HISTOGRAM_WRAPPER_CUH_
#define _STP_SINGLE_HISTOGRAM_WRAPPER_CUH_

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif//__CUDACC_RTC__

//Histogram
#include "../STPSingleHistogram.hpp"

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPSingleHistogramWrapper is a high-level wrapper class over data structure STPSingleHistogram,
	 * it provides easier data access to histogram for every pixel.
	*/
	class STPSingleHistogramWrapper {
	private:

		//Constant reference to STPBin
		typedef const STPSingleHistogram::STPBin& STPSingleHistogramBin_ct;

		const STPSingleHistogram& Histogram;

		/**
		 * @brief Get the index to the STPBin, corresponds to the first STPBin for the histogram of this pixel index.
		 * The range of bin for the current histogram is [pixel_index, pixel_index + 1)
		 * @see STPSingleHistogram
		 * @param pixel_index The index of the pixel
		 * @return The index to the histogram for this pixel in the array of STPBin
		*/
		__device__ unsigned int operator[](unsigned int) const;

		/**
		 * @brief Get the pointer to the bin in the bin array
		 * @param bin_index The index of the bin
		 * @return The const pointer to the bin
		*/
		__device__ STPSingleHistogramBin_ct getBin(unsigned int) const;

		/**
		 * @brief Get the item the bin belongs to
		 * @param bin The bin to be retrieved
		 * @return The item
		*/
		__device__ STPDiversity::Sample getItem(STPSingleHistogramBin_ct) const;

		/**
		 * @brief Get the weight of the item in this bin for the histogram this bin is in.
		 * Weight is a float value between 0.0(exclusive) and 1.0(inclusive)
		 * @param bin The bin to be retrieved
		 * @return The weight of this item
		*/
		__device__ float getWeight(STPSingleHistogramBin_ct) const;

	public:

		/**
		 * @brief Init single histogram wrapper
		 * @param histogram The histogram data to be operated on
		*/
		__device__ STPSingleHistogramWrapper(const STPSingleHistogram&);

		__device__ STPSingleHistogramWrapper(const STPSingleHistogramWrapper&) = delete;

		__device__ STPSingleHistogramWrapper(STPSingleHistogramWrapper&&) = delete;

		__device__ STPSingleHistogramWrapper& operator=(const STPSingleHistogramWrapper&) = delete;

		__device__ STPSingleHistogramWrapper& operator=(STPSingleHistogramWrapper&&) = delete;

		__device__ ~STPSingleHistogramWrapper();

		/**
		 * @brief Get the histogram for the current pixel index, perform user-defined operation on the pixel using the single histogram for this pixel.
		 * Function will loop over all bins in the histogram, in each iteration function provided by user will be called.
		 * @param pixel_index The histogram which will be retrieved for the pixel.
		 * @param function The user-defined function. The function should have the following signature:
		 * void(I item, float weight)
		 * where item (currently only STPDiversity::Sample is supported) is the item in the current bin and
		 * weight is the percentage of item occupied in the histogram, the sum of weight of all items in any the same histogram is always 1.0.
		 * Weight will always be in the range (0.0f, 1.0f]
		*/
		template<class Func>
		__device__ __inline__ void operator()(unsigned int, Func&&) const;

	};

}
#include "STPSingleHistogramWrapper.inl"
#endif//_STP_SINGLE_HISTOGRAM_WRAPPER_CUH_