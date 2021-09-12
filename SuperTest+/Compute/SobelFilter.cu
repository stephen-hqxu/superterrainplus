#pragma once
//Filter optimisation mode
#include "SobelFilter.cuh"

#include <device_launch_parameters.h>

#include <glm/geometric.hpp>

using glm::ivec2;
using glm::uvec2;
using glm::vec2;

__constant__ uvec2 ImageDimension[1];

//Declaration

__device__ float calcGradient(float*, uvec2, uvec2);

__device__ __forceinline__ uvec2 clampIndex(ivec2, uvec2);

//Definition

#define SETUP() const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y; \
if (x >= ImageDimension->x || y >= ImageDimension->y) { \
	return; \
} \
const unsigned int index = x + y * ImageDimension->x

#define SOBEL_FILTER(M) template<> \
__global__ void sobelFilter<M>(float* filterOut, float* imageIn)

#define CACHE_SETUP() const uvec2 block = uvec2(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y), \
local_thread = uvec2(threadIdx.x, threadIdx.y); \
const unsigned int threadperblock = blockDim.x * blockDim.y; \
\
extern __shared__ float Cache[]; \
const uvec2 cacheSize = uvec2(blockDim.x, blockDim.y) + 2u; \
unsigned int iteration = 0u; \
const unsigned int cacheSize_total = cacheSize.x * cacheSize.y

SOBEL_FILTER(SobelFilterMode::VanillaConvolution) {
	SETUP();
	
	//vanilla convolution uses no cache
	filterOut[index] = calcGradient(imageIn, uvec2(x, y), *ImageDimension);
	
}

SOBEL_FILTER(SobelFilterMode::RemappedCacheLoadConvolution) {
	SETUP();

	//load image in block into cache
	//this is exactly the same as the current model used during rendering buffer generation in heightfield generator
	CACHE_SETUP();

	while (iteration < cacheSize_total) {
		const unsigned int cacheIdx = (threadIdx.x + blockDim.x * threadIdx.y) + iteration;
		const uvec2 worker = block + uvec2(cacheIdx % cacheSize.x, cacheIdx / cacheSize.x);
		const uvec2 clamppeWorkerIdx = clampIndex(static_cast<ivec2>(worker) - 1, *ImageDimension);
		const unsigned int workerIdx = clamppeWorkerIdx.x + clamppeWorkerIdx.y * ImageDimension->x;

		if (cacheIdx < cacheSize_total) {
			Cache[cacheIdx] = imageIn[workerIdx];
		}
		iteration += threadperblock;
	}
	__syncthreads();

	//in the actual code we don't compute the pixels at the edge, here we just make things simple and do it for every pixel
	//compute
	filterOut[index] = calcGradient(Cache, local_thread + 1u, cacheSize);
}

SOBEL_FILTER(SobelFilterMode::CoalescedCacheLoadConvolution) {
	SETUP();

	//load the cache
	//this is a purposed version of cache loading, it prioritises coalesced memory access
	CACHE_SETUP();
	//load the central first
	const uvec2 cacheCoord = local_thread + 1u;
	{
		const unsigned int cacheIdx = cacheCoord.x + cacheCoord.y * cacheSize.x;
		Cache[cacheIdx] = imageIn[index];
	}
	//load the border
	while (iteration < cacheSize_total) {
		const unsigned int cacheIdx = (threadIdx.x + blockDim.x * threadIdx.y) + iteration;
		const uvec2 cacheCoord = uvec2(cacheIdx % cacheSize.x, cacheIdx / cacheSize.x);
		const uvec2 worker = block + cacheCoord;
		const uvec2 clamppeWorkerIdx = clampIndex(static_cast<ivec2>(worker) - 1, *ImageDimension);
		const unsigned int workerIdx = clamppeWorkerIdx.x + clamppeWorkerIdx.y * ImageDimension->x;

		if ((cacheCoord.x == 0u || cacheCoord.y == 0u || cacheCoord.x == cacheSize.x - 1u || cacheCoord.y == cacheSize.y - 1u) && cacheIdx < cacheSize_total) {
			Cache[cacheIdx] = imageIn[workerIdx];
		}
		iteration += threadperblock;
	}

	__syncthreads();

	//compute
	filterOut[index] = calcGradient(Cache, cacheCoord, cacheSize);
	
}

__device__ float calcGradient(float* image, uvec2 coord, uvec2 dimension) {
	const auto clamppedLoad = [image, coord, dimension]__device__(ivec2 offset) -> float {
		const uvec2 clamppedCoord = clampIndex(static_cast<ivec2>(coord) + offset, dimension);
		return image[clamppedCoord.x + clamppedCoord.y * dimension.x];
	};

	//load cells
	float cell[8];
	cell[0] = clamppedLoad(ivec2(-1, -1));
	cell[1] = clamppedLoad(ivec2(0, -1));
	cell[2] = clamppedLoad(ivec2(+1, -1));
	cell[3] = clamppedLoad(ivec2(-1, 0));
	cell[4] = clamppedLoad(ivec2(+1, 0));
	cell[5] = clamppedLoad(ivec2(-1, +1));
	cell[6] = clamppedLoad(ivec2(0, +1));
	cell[7] = clamppedLoad(ivec2(+1, +1));

	//calculate horizontal and vertical derivatives
	const vec2 Grad = vec2(
		cell[0] + 2 * cell[3] + cell[5] - (cell[2] + 2 * cell[4] + cell[7]),
		cell[0] + 2 * cell[1] + cell[2] - (cell[5] + 2 * cell[6] + cell[7])
	);
	//calculate output gradient
	return glm::length(Grad);
}

__host__ const uvec2* getDimensionSymbol() {
	return ImageDimension;
}

__device__ __forceinline__ uvec2 clampIndex(ivec2 index, uvec2 dimension) {
	return static_cast<uvec2>(glm::clamp(index, ivec2(0), static_cast<ivec2>(dimension - 1u)));
}