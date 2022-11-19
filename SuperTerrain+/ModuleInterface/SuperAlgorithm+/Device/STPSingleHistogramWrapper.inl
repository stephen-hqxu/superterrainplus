//Contain inline definitions for template functions in STPSingleHistogramWrapper
//THIS INLINE FILE IS MANAGED AUTOMATICALLY, PLEASE DON'T INCLUDE IT
#ifdef _STP_SINGLE_HISTOGRAM_WRAPPER_CUH_

template<class Func>
__device__ inline void SuperTerrainPlus::STPAlgorithm::STPSingleHistogramWrapper::iterate(
	const STPSingleHistogram& histogram, const unsigned int pixel_index, Func&& function) {
	const auto [bin, start_offset] = histogram;
	//get the bin index range for the current histogram
	const unsigned int begin = start_offset[pixel_index],
		end = start_offset[pixel_index + 1u];

	//loop through every bin
	for (unsigned int bin_idx = begin; bin_idx < end; bin_idx++) {
		//get the pointer to the current bin
		const STPSingleHistogram::STPBin& curr_bin = bin[bin_idx];
		//call user defined function
		std::forward<Func>(function)(curr_bin.Item, curr_bin.Data.Weight);
	}
}

#endif//_STP_SINGLE_HISTOGRAM_WRAPPER_CUH_