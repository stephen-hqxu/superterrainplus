//Contain inline definitions for template functions in STPSingleHistogramWrapper

//THIS INLINE FILE IS MANAGED AUTOMATICALLY, PLEASE DON'T INCLUDE IT
#ifdef _STP_SINGLE_HISTOGRAM_WRAPPER_CUH_

template<class Func>
__device__ __inline__ void SuperTerrainPlus::STPCompute::STPSingleHistogramWrapper::operator()(unsigned int pixel_index, Func&& function) const {
	//get the bin index range for the current histogram
	const unsigned int begin = (*this)[pixel_index],
		end = (*this)[pixel_index + 1];

	//loop through every bin
	for (unsigned int bin_index = begin; bin_index < end; bin_index++) {
		//get the pointer to the current bin
		STPSingleHistogramBin_ct bin = this->getBin(bin_index);
		//call user defined function
		std::forward<Func>(function)(this->getItem(bin), this->getWeight(bin));
	}
}

#endif//_STP_SINGLE_HISTOGRAM_WRAPPER_CUH_