#ifndef _STP_COMMON_GENERATOR_CUH_
#define _STP_COMMON_GENERATOR_CUH_

//Device Library
#include <STPSimplexNoise.cuh>
#include <STPKernelMath.cuh>

#ifndef __CUDACC_RTC__
#error __FILE__ can only be compiled in NVRTC
#endif//__CUDACC_RTC__

namespace STPCommonGenerator {

	//This is the dimension of map of one chunk
	extern __constant__ uint2 Dimension[1];
	//Dimension / 2
	extern __constant__ float2 HalfDimension[1];
	//This is the dimension of map in the entire rendered chunk
	extern __constant__ uint2 RenderedDimension[1];

	extern __constant__ SuperTerrainPlus::STPAlgorithm::STPPermutation Permutation[1];
}

#endif//_STP_COMMON_GENERATOR_CUH_