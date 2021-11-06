//THIS IS AN INLINED TEMPLATE DEFINITINO FOR KERNEL MATH, DO NOT INCLUDE IT SEPARATELY

#ifdef _STP_KERNEL_MATH_CUH_

template<class It, typename T>
__device__ const It* SuperTerrainPlus::STPCompute::STPKernelMath::lower_bound(const It* first, const It* last, T value) {
	static auto less_than = []__device__(const T& current, const T& value) constexpr -> bool {
		return current < value;
	};
	return STPKernelMath::lower_bound(first, last, value, less_than);
}

template<class It, typename T, class Comp>
__device__ const It* SuperTerrainPlus::STPCompute::STPKernelMath::lower_bound(const It* first, const It* last, T value, Comp comparator) {
	//std::lower_bound implementation
	const It* it;
	//distance
	int count = last - first, step;
	while (count > 0) {
		it = first;
		step = count / 2;
		//advance
		it += step;
		if (comparator(*it, value)) {
			first = ++it;
			count -= step + 1;
		}
		else {
			count = step;
		}
	}
	return first;
}
#endif//_STP_KERNEL_MATH_CUH_