//DEFINITIONS FOR VERIOUS DEVICE FUNCTION UTILITIES
#ifdef _STP_DEVICE_LAUNCH_SETUP_CUH_

#ifndef __CUDACC_RTC__
#include "../Utility/STPDeviceErrorHandler.hpp"

//Vector
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <device_launch_parameters.h>
#include <vector_types.h>

#include <type_traits>
#endif

#define DEV_LAUNCH_NAME SuperTerrainPlus::STPDeviceLaunchSetup

#ifndef __CUDACC_RTC__
template<DEV_LAUNCH_NAME::STPDimensionSize Block, class Vec>
__host__ inline DEV_LAUNCH_NAME::STPLaunchConfiguration DEV_LAUNCH_NAME::determineLaunchConfiguration(
	const int blockSize, const Vec& threadSize) {
	constexpr static auto getGridDim = []() constexpr noexcept -> STPDimensionSize {
		if constexpr (std::is_integral_v<Vec>) {
			return 1u;
		} else {
			return static_cast<STPDimensionSize>(Vec::length());
		}
	};
	constexpr static STPDimensionSize Grid = getGridDim();
	//TODO: again, requires clause in C++ 20
	static_assert(Grid >= 1u && Grid <= 3u && Block >= 1u && Block <= 3u,
		"The dimension of grid/block can only be between 1 and 3");

	using glm::uvec2, glm::uvec3;
	/* ------------------------------- Determine Block Size ----------------------- */
	uvec3 dimBlock;
	if constexpr (Block == 1u) {
		dimBlock = uvec3(blockSize, uvec2(1u));
	} else {
		//the block size returned by CUDA occupancy calculator must be a multiple of warp size
		//because the API suggests the block size is *optimal*
		dimBlock = uvec3(blockSize / STPDeviceLaunchSetup::WarpSize, STPDeviceLaunchSetup::WarpSize, 1u);
		if constexpr (Block == 3u) {
			//as of current the warp size is divisible like this; don't know if any future GPU will be changed
			dimBlock.y = STPDeviceLaunchSetup::WarpSize / 4u;
			dimBlock.z = 4u;
		}
	}

	/* -------------------------------- Determine Grid Size ---------------------- */
	//expand thread size to a 3D vector
	uvec3 dimThread;
	//pad the extra, unused components with one
	if constexpr (Grid == 1u) {
		dimThread = uvec3(threadSize, uvec2(1u));
	} else if constexpr (Grid == 2u) {
		dimThread = uvec3(threadSize, 1u);
	} else {
		dimThread = uvec3(threadSize);
	}
	//round up the grid size
	const uvec3 dimGrid = (dimThread + dimBlock - 1u) / dimBlock;

	return std::make_tuple(
		dim3(dimGrid.x, dimGrid.y, dimGrid.z),
		dim3(dimBlock.x, dimBlock.y, dimBlock.z)
	);
}

template<DEV_LAUNCH_NAME::STPDimensionSize Block, class Vec, class Func>
__host__ inline DEV_LAUNCH_NAME::STPLaunchConfiguration DEV_LAUNCH_NAME::determineLaunchConfiguration(
	const Func func, const Vec& threadSize, const size_t dynamicSMemSize, const int blockSizeLimit) {
	int minGridSize, blockSize;
	STP_CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, dynamicSMemSize, blockSizeLimit));
	(void)minGridSize;

	//delegate the call to the overload
	return STPDeviceLaunchSetup::determineLaunchConfiguration<Block>(blockSize, threadSize);
}
#endif//__CUDACC_RTC__

#ifdef __CUDACC__
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