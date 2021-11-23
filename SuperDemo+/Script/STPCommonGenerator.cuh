#ifndef _STP_COMMON_GENERATOR_CUH_
#define _STP_COMMON_GENERATOR_CUH_

#ifndef __CUDACC_RTC__
#error __FILE__ can only be compiled in NVRTC
#endif//__CUDACC_RTC__

namespace STPCommonGenerator {

	/**
	 * @brief Get the dimension of a texture map.
	 * @return The 2D dimension of a single texture map, in x, z direction.
	*/
	__device__ const uint2& mapDimension();

	/**
	 * @brief Get the half dimension of a texture map.
	 * @return The 2D half dimension of a single texture map. It's equalvalent to mapDimension() / 2.
	*/
	__device__ const float2& mapDimensionHalf();

	/**
	 * @brief Get the rendered dimension of a texture map.
	 * @return The 2D rendered dimension of a single texture map. It's equivalent to mapDimension() * RenderChunk.xy.
	*/
	__device__ const uint2& mapDimensionRendered();
}

#endif//_STP_COMMON_GENERATOR_CUH_