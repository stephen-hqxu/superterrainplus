#pragma once
#ifndef _STP_SINGLE_HISTOGRAM_WRAPPER_CUH_
#define _STP_SINGLE_HISTOGRAM_WRAPPER_CUH_

//Histogram
#include "../STPSingleHistogram.hpp"

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPSingleHistogramWrapper is a high-level wrapper class over data structure STPSingleHistogram,
	 * it provides easier data access to histogram for every pixel.
	*/
	namespace STPSingleHistogramWrapper {

		/**
		 * @brief Get the histogram for the current pixel index, perform user-defined operation on the pixel using the single histogram for this pixel.
		 * Function will loop over all bins in the histogram, in each iteration function provided by user will be called.
		 * @param histogram The histogram data to be operated on.
		 * @param pixel_index The histogram which will be retrieved for the pixel.
		 * @param function The user-defined function. The function should have the following signature:
		 * void(I item, float weight)
		 * where item (currently only STPDiversity::Sample is supported) is the item in the current bin and
		 * weight is the percentage of item occupied in the histogram, the sum of weight of all items in any the same histogram is always 1.0.
		 * Weight will always be in the range (0.0f, 1.0f]
		*/
		template<class Func>
		__device__ void iterate(const STPSingleHistogram&, unsigned int, Func&&);

	}

}
#include "STPSingleHistogramWrapper.inl"
#endif//_STP_SINGLE_HISTOGRAM_WRAPPER_CUH_