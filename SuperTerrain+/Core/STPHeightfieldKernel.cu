#include <SuperTerrain+/GPGPU/STPHeightfieldKernel.cuh>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

/* --------- Kernel Declaration ----------- */

using namespace SuperTerrainPlus;

#include <glm/geometric.hpp>

using glm::uvec2;

__global__ static void curandInitKERNEL(STPHeightfieldKernel::STPcurand_t*, unsigned long long, unsigned int);

__global__ static void hydraulicErosionKERNEL(
	float*, const STPEnvironment::STPRainDropSetting*, STPNearestNeighbourInformation, STPErosionBrush, STPHeightfieldKernel::STPcurand_t*);

__global__ static void texture32Fto16KERNEL(float*, unsigned short*, uvec2);

__host__ STPHeightfieldKernel::STPcurand_arr STPHeightfieldKernel::curandInit(const unsigned long long seed, const unsigned int count, const cudaStream_t stream) {
	//determine launch parameters
	int Mingridsize, gridsize, blocksize;
	STP_CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &curandInitKERNEL));
	gridsize = (count + blocksize - 1) / blocksize;

	//allocating spaces for rng storage array
	//because RNG storage is persistent, i.e., we will be keep reusing it once allocated, no need to allocate from a memory pool.
	STPcurand_arr rng = STPSmartDeviceMemory::makeDevice<STPcurand_t[]>(count);
	//and send to kernel to init rng sequences
	curandInitKERNEL<<<gridsize, blocksize, 0, stream>>>(rng.get(), seed, count);
	STP_CHECK_CUDA(cudaGetLastError());

	return rng;
}

__host__ void STPHeightfieldKernel::hydraulicErosion(float* const heightmap_storage,
	const STPEnvironment::STPRainDropSetting* const raindrop_setting, const STPNearestNeighbourInformation& nn_info,
	const STPErosionBrush& brush, const unsigned int raindrop_count, STPcurand_t* const rng, const cudaStream_t stream) {
	//brush contains two components: weights (float) and indices (int)
	const unsigned int erosionBrushCache_size = brush.BrushSize * (sizeof(int) + sizeof(float));
	//launch para
	int Mingridsize, gridsize, blocksize;
	STP_CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &hydraulicErosionKERNEL, erosionBrushCache_size));
	gridsize = (raindrop_count + blocksize - 1) / blocksize;

	//erode the heightmap
	hydraulicErosionKERNEL<<<gridsize, blocksize, erosionBrushCache_size, stream>>>(
		heightmap_storage, raindrop_setting, nn_info, brush, rng);
	STP_CHECK_CUDA(cudaGetLastError());
}

__host__ void STPHeightfieldKernel::texture32Fto16(float* const input, unsigned short* const output,
	const uvec2 dimension, const cudaStream_t stream) {
	int Mingridsize, blocksize;
	STP_CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &texture32Fto16KERNEL));
	const uvec2 Dimblocksize(32u, static_cast<unsigned int>(blocksize) / 32u),
		Dimgridsize = (dimension + Dimblocksize - 1u) / Dimblocksize;

	//compute
	texture32Fto16KERNEL<<<dim3(Dimgridsize.x, Dimgridsize.y), dim3(Dimblocksize.x, Dimblocksize.y), 0, stream>>>(
		input, output, dimension);
	STP_CHECK_CUDA(cudaGetLastError());
}

/* --------- Kernel Definition ----------- */

#include <device_launch_parameters.h>

using glm::vec2;

__global__ void curandInitKERNEL(STPHeightfieldKernel::STPcurand_t* const rng, const unsigned long long seed, const unsigned int count) {
	//current working index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= count) {
		return;
	}

	//the same seed but we are looking for different sequence
	curand_init(seed, static_cast<unsigned long long>(index), 0, &rng[index]);
}

//It's raining
#include <SuperTerrain+/GPGPU/STPRainDrop.cuh>

__global__ void hydraulicErosionKERNEL(float* const heightmap_storage,
	const STPEnvironment::STPRainDropSetting* const raindrop_setting, const STPNearestNeighbourInformation nn_info,
	const STPErosionBrush brush, STPHeightfieldKernel::STPcurand_t* const rng) {
	//current working index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= raindrop_setting->RainDropCount) {
		return;
	}

	//convert to (base, dimension - 1]
	//range: dimension
	//Generate the raindrop at the central chunk only
	__shared__ uvec2 base;
	__shared__ uvec2 range;
	if (threadIdx.x == 0u) {
		base = nn_info.MapSize - 1u,
			range = (nn_info.ChunkNearestNeighbour / 2u) * nn_info.MapSize;
	}
	__syncthreads();

	//generating random location
	//first we generate the number (0.0f, 1.0f]
	vec2 initPos = { };
	initPos.x = curand_uniform(&rng[index]);
	initPos.y = curand_uniform(&rng[index]);
	//range conversion
	initPos *= base;
	initPos += range;

	//spawn the raindrop
	STPRainDrop droplet(initPos, raindrop_setting->initWaterVolume, raindrop_setting->initSpeed, nn_info.TotalMapSize);
	droplet(heightmap_storage, static_cast<const STPEnvironment::STPRainDropSetting&>(*raindrop_setting), brush);
}

#include <limits>
constexpr static unsigned short FP32toUINT16constant = std::numeric_limits<unsigned short>::max();

__global__ void texture32Fto16KERNEL(float* const input, unsigned short* const output, const uvec2 dimension) {
	//the current working pixel
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y,
		index = x + y * dimension.x;
	//range check
	if (x >= dimension.x || y >= dimension.y) {
		return;
	}

	output[index] = static_cast<unsigned short>(rintf(__saturatef(input[index]) * FP32toUINT16constant));
}