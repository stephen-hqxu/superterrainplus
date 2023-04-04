//DEFINITIONS FOR VERIOUS DEVICE FUNCTION UTILITIES
#ifdef _STP_DEVICE_LAUNCH_SETUP_CUH_

#ifndef __CUDACC_RTC__
#include "../Utility/STPDeviceErrorHandler.hpp"

//Vector
#include <glm/common.hpp>
#include <glm/vec2.hpp>

//Standard Library
#include <type_traits>
#endif

#define DEV_LAUNCH_NAME SuperTerrainPlus::STPDeviceLaunchSetup

#ifndef __CUDACC_RTC__
template<class Vec>
__host__ inline glm::uvec3 DEV_LAUNCH_NAME::STPInternal::normaliseThreadSize(Vec threadSize) {
	constexpr static auto getThreadDim = []() constexpr noexcept -> STPDimensionSize {
		if constexpr (std::is_integral_v<Vec>) {
			return 1u;
		} else {
			return static_cast<STPDimensionSize>(Vec::length());
		}
	};
	constexpr static STPDimensionSize ThreadDimension = getThreadDim();

	//make all components at least 1 in case it contains a value less than one
	threadSize = glm::max(threadSize, 1u);
	
	using glm::uvec2, glm::uvec3;
	//expand thread size to a 3D vector
	uvec3 dimThread;
	//pad the extra, unused components with one
	//truncate extra components if thread dimension is more than 3
	if constexpr (ThreadDimension == 1u) {
		dimThread = uvec3(threadSize, uvec2(1u));
	} else if constexpr (ThreadDimension == 2u) {
		dimThread = uvec3(threadSize, 1u);
	} else {
		dimThread = static_cast<uvec3>(threadSize);
	}
	return dimThread;
}

template<DEV_LAUNCH_NAME::STPDimensionSize Block, class Vec>
__host__ inline DEV_LAUNCH_NAME::STPLaunchConfiguration DEV_LAUNCH_NAME::determineLaunchConfiguration(
	const int blockSize, const Vec& threadSize) {
	//TODO: use requires clause in C++ 20
	static_assert(Block >= 1u && Block <= 3u, "The dimension of grid/block can only be between 1 and 3");

	return STPInternal::determineLaunchConfiguration(Block,
		static_cast<unsigned int>(blockSize), STPInternal::normaliseThreadSize(threadSize));
}

template<DEV_LAUNCH_NAME::STPDimensionSize Block, class Vec, class Func>
__host__ inline DEV_LAUNCH_NAME::STPLaunchConfiguration DEV_LAUNCH_NAME::determineLaunchConfiguration(
	const Func func, const Vec& threadSize, const size_t dynamicSMemSize, const int blockSizeLimit) {
	int minGridSize, blockSize;
	STP_CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, dynamicSMemSize, blockSizeLimit));
	(void)minGridSize;

	return STPDeviceLaunchSetup::determineLaunchConfiguration<Block>(blockSize, threadSize);
}
#endif//__CUDACC_RTC__

#ifdef __CUDACC__
#ifndef __CUDACC_RTC__
#include <device_launch_parameters.h>
#include <vector_types.h>
#endif//__CUDACC_RTC__

template<DEV_LAUNCH_NAME::STPDimensionSize Dim>
__device__ inline auto DEV_LAUNCH_NAME::calcThreadIndex() {
	//TODO: use requires clause in C++ 20
	static_assert(Dim >= 1u && Dim <= 3u, "The dimension of thread can only be between 1 and 3");
	
	//CUDA vector type allows us to use structured binding, but GLM vector does not
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if constexpr (Dim == 1u) {
		return make_uint1(x);
	} else {
		const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		if constexpr (Dim == 2u) {
			return make_uint2(x, y);
		} else {
			const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
			return make_uint3(x, y, z);
		}
	}
}
#endif//__CUDACC__

#undef DEV_LAUNCH_NAME

#endif//_STP_DEVICE_LAUNCH_SETUP_CUH_