#include <SuperTerrain+/GPGPU/STPHeightfieldKernel.cuh>

#include <SuperTerrain+/Utility/STPSmartDeviceMemory.tpp>
//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

/* --------- Kernel Declaration ----------- */

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPCompute;

#include <glm/geometric.hpp>

using glm::uvec2;

__global__ static void curandInitKERNEL(STPHeightfieldKernel::STPcurand_t*, unsigned long long, unsigned int);

__global__ static void initGlobalLocalIndexKERNEL(unsigned int*, uvec2, uvec2, uvec2);

__global__ static void hydraulicErosionKERNEL(STPFreeSlipFloatManager, const STPEnvironment::STPHeightfieldSetting*, STPHeightfieldKernel::STPcurand_t*);

__global__ static void texture32Fto16KERNEL(float*, unsigned short*, uvec2);

__host__ STPHeightfieldKernel::STPcurand_arr STPHeightfieldKernel::curandInit(unsigned long long seed, unsigned int count) {
	//determine launch parameters
	int Mingridsize, gridsize, blocksize;
	STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &curandInitKERNEL));
	gridsize = (count + blocksize - 1) / blocksize;

	//allocating spaces for rng storage array
	STPcurand_arr rng = STPSmartDeviceMemory::makeDevice<STPcurand_t[]>(count);
	//and send to kernel to init rng sequences
	curandInitKERNEL << <gridsize, blocksize >> > (rng.get(), seed, count);
	STPcudaCheckErr(cudaGetLastError());

	return rng;
}

__host__ STPHeightfieldKernel::STPIndexTable STPHeightfieldKernel::initGlobalLocalIndex(uvec2 chunkRange, uvec2 tableSize, uvec2 mapSize) {
	const size_t index_count = tableSize.x * tableSize.y;
	//launch parameters
	int Mingridsize, blocksize;
	STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &initGlobalLocalIndexKERNEL));
	const uvec2 Dimblocksize(32u, static_cast<unsigned int>(blocksize) / 32u),
		Dimgridsize = (tableSize + Dimblocksize - 1u) / Dimblocksize;

	//allocation
	STPIndexTable indexTable = STPSmartDeviceMemory::makeDevice<unsigned int[]>(index_count);
	//compute
	initGlobalLocalIndexKERNEL << <dim3(Dimgridsize.x, Dimgridsize.y), dim3(Dimblocksize.x, Dimblocksize.y) >> > (
		indexTable.get(), chunkRange, tableSize, mapSize);
	STPcudaCheckErr(cudaGetLastError());

	return indexTable;
}

__host__ void STPHeightfieldKernel::hydraulicErosion
	(STPFreeSlipFloatManager heightmap_storage, const STPEnvironment::STPHeightfieldSetting* heightfield_settings, 
		unsigned int brush_size, unsigned int raindrop_count, STPcurand_t* rng, cudaStream_t stream) {
	//brush contains two components: weights (float) and indices (int)
	const unsigned int erosionBrushCache_size = brush_size * (sizeof(int) + sizeof(float));
	//launc para
	int Mingridsize, gridsize, blocksize;
	STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &hydraulicErosionKERNEL, erosionBrushCache_size));
	gridsize = (raindrop_count + blocksize - 1) / blocksize;

	//erode the heightmap
	hydraulicErosionKERNEL << <gridsize, blocksize, erosionBrushCache_size, stream >> > (heightmap_storage, heightfield_settings, rng);
	STPcudaCheckErr(cudaGetLastError());
}

__host__ void STPHeightfieldKernel::texture32Fto16(float* input, unsigned short* output, uvec2 dimension, unsigned int channel, cudaStream_t stream) {
	const uvec2 totalDimension = dimension * channel;

	int Mingridsize, blocksize;
	STPcudaCheckErr(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &texture32Fto16KERNEL));
	const uvec2 Dimblocksize(32u, static_cast<unsigned int>(blocksize) / 32u),
		Dimgridsize = (totalDimension + Dimblocksize - 1u) / Dimblocksize;

	//compute
	texture32Fto16KERNEL << <dim3(Dimgridsize.x, Dimgridsize.y), dim3(Dimblocksize.x, Dimblocksize.y), 0, stream >> > (input, output, totalDimension);
	STPcudaCheckErr(cudaGetLastError());
}

/* --------- Kernel Definition ----------- */

#include <device_launch_parameters.h>

using glm::vec2;

__global__ void curandInitKERNEL(STPHeightfieldKernel::STPcurand_t* rng, unsigned long long seed, unsigned int count) {
	//current working index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= count) {
		return;
	}

	//the same seed but we are looking for different sequence
	curand_init(seed, static_cast<unsigned long long>(index), 0, &rng[index]);
}

__global__ void initGlobalLocalIndexKERNEL(unsigned int* output, uvec2 chunkRange, uvec2 tableSize, uvec2 mapSize) {
	//current pixel
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y,
		rowCount = tableSize.x,
		globalidx = x + y * rowCount;
	if (x >= tableSize.x || y >= tableSize.y) {
		return;
	}

	//simple maths
	const uvec2 globalPos = uvec2(globalidx - y * rowCount, y);
	const uvec2 chunkPos = globalPos / mapSize;//non-negative integer division is a floor
	const uvec2 localPos = globalPos - chunkPos * mapSize;

	output[globalidx] = (chunkPos.x + chunkRange.x * chunkPos.y) * mapSize.x * mapSize.y + (localPos.x + mapSize.x * localPos.y);
}

//It's raining
#include <SuperTerrain+/GPGPU/STPRainDrop.cuh>

__global__ void hydraulicErosionKERNEL
	(STPFreeSlipFloatManager heightmap_storage, const STPEnvironment::STPHeightfieldSetting* heightfield_settings, STPHeightfieldKernel::STPcurand_t* rng) {
	//current working index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= heightfield_settings->RainDropCount) {
		return;
	}

	//convert to (base, dimension - 1]
	//range: dimension
	//Generate the raindrop at the central chunk only
	__shared__ uvec2 base;
	__shared__ uvec2 range;
	if (threadIdx.x == 0u) {
		const uvec2& dimension = heightmap_storage.Data->Dimension;

		base = dimension - 1u,
			range = (heightmap_storage.Data->FreeSlipChunk / 2u) * dimension;
	}
	__syncthreads();

	//generating random location
	//first we generate the number (0.0f, 1.0f]
	vec2 initPos = vec2(curand_uniform(&rng[index]), curand_uniform(&rng[index]));
	//range convertion
	initPos *= base;
	initPos += range;

	//spawn the raindrop
	STPRainDrop droplet(initPos, heightfield_settings->initWaterVolume, heightfield_settings->initSpeed);
	droplet.Erode(static_cast<const STPEnvironment::STPRainDropSetting*>(heightfield_settings), heightmap_storage);
}

#include <limits>
constexpr static unsigned short FP32toUINT16constant = std::numeric_limits<unsigned short>::max();

__global__ void texture32Fto16KERNEL(float* input, unsigned short* output, uvec2 dimension) {
	//the current working pixel
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y,
		index = x + y * dimension.x;
	//range check
	if (x >= dimension.x || y >= dimension.y) {
		return;
	}

	output[index] = static_cast<unsigned short>(glm::clamp(input[index], 0.0f, 1.0f) * FP32toUINT16constant);
}