#pragma once
#ifndef _STP_WORLD_MAP_PIXEL_FORMAT_HPP_
#define _STP_WORLD_MAP_PIXEL_FORMAT_HPP_

#include <cuda/std/cstdint>

namespace SuperTerrainPlus {

	/* this header can be used safely in NVRTC, provided CUDA include directory is added to the compiler option explicitly */

	//biomemap
	typedef cuda::std::uint16_t STPSample_t;
	//single-precision floating-point heightmap
	typedef float STPHeightFloat_t;
	//16-bit unsigned integer fixed-point heightmap
	typedef cuda::std::uint16_t STPHeightFixed_t;
	//splatmap
	typedef cuda::std::uint8_t STPRegion_t;

	//seed for any random number generator
	typedef cuda::std::uint64_t STPSeed_t;

}

#endif//_STP_WORLD_MAP_PIXEL_FORMAT_HPP_