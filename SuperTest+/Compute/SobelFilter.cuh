#pragma once
#ifndef _SobelFilter_CUH_
#define _SobelFilter_CUH_

#include <cuda_runtime.h>

#include <glm/vec2.hpp>

enum class SobelFilterMode : unsigned char {
	VanillaConvolution = 0x00u,
	RemappedCacheLoadConvolution = 0x01u,
	CoalescedCacheLoadConvolution = 0x02u
};

template<SobelFilterMode M>
__global__ void sobelFilter(float*, float*);

__host__ const glm::uvec2* getDimensionSymbol();

#endif//_SobelFilter_CUH_