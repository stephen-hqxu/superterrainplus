#include <SuperTerrain+/GPGPU/STPHeightfieldKernel.cuh>

#include <SuperTerrain+/Utility/STPDeviceLaunchSetup.cuh>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

/* --------- Kernel Declaration ----------- */

using namespace SuperTerrainPlus;

#include <glm/geometric.hpp>

using glm::uvec2;

__global__ static void curandInitKERNEL(STPHeightfieldKernel::STPcurand_t*, STPSeed_t, unsigned int);

__global__ static void hydraulicErosionKERNEL(STPHeightFloat_t*, const STPEnvironment::STPRainDropSetting*,
	STPNearestNeighbourInformation, STPErosionBrush, STPHeightfieldKernel::STPcurand_t*);

__global__ static void formatHeightmapKERNEL(STPHeightFloat_t*, STPHeightFixed_t*, uvec2);

__host__ STPHeightfieldKernel::STPcurand_arr STPHeightfieldKernel::curandInit(
	const STPSeed_t seed, const unsigned int count, const cudaStream_t stream) {
	const auto [gridSize, blockSize] = STPDeviceLaunchSetup::determineLaunchConfiguration<1u>(curandInitKERNEL, count);

	//allocating spaces for rng storage array
	//because RNG storage is persistent, i.e., we will be keep reusing it once allocated, no need to allocate from a memory pool.
	STPcurand_arr rng = STPSmartDeviceMemory::makeDevice<STPcurand_t[]>(count);
	//and send to kernel to init rng sequences
	curandInitKERNEL<<<gridSize, blockSize, 0, stream>>>(rng.get(), seed, count);
	STP_CHECK_CUDA(cudaGetLastError());

	return rng;
}

__host__ void STPHeightfieldKernel::hydraulicErosion(STPHeightFloat_t* const heightmap_storage,
	const STPEnvironment::STPRainDropSetting* const raindrop_setting, const STPNearestNeighbourInformation& nn_info,
	const STPErosionBrush& brush, const unsigned int raindrop_count, STPcurand_t* const rng, const cudaStream_t stream) {
	//brush contains two components: weights (float) and indices (int)
	const unsigned int erosionBrushCache_size = brush.BrushSize * (sizeof(int) + sizeof(float));
	const auto [gridSize, blockSize] = STPDeviceLaunchSetup::determineLaunchConfiguration<1u>(hydraulicErosionKERNEL,
		raindrop_count, erosionBrushCache_size);

	hydraulicErosionKERNEL<<<gridSize, blockSize, erosionBrushCache_size, stream>>>(
		heightmap_storage, raindrop_setting, nn_info, brush, rng);
	STP_CHECK_CUDA(cudaGetLastError());
}

__host__ void STPHeightfieldKernel::formatHeightmap(STPHeightFloat_t* const input, STPHeightFixed_t* const output,
	const uvec2 dimension, const cudaStream_t stream) {
	const auto [gridSize, blockSize] = STPDeviceLaunchSetup::determineLaunchConfiguration<2u>(formatHeightmapKERNEL, dimension);

	formatHeightmapKERNEL<<<gridSize, blockSize, 0, stream>>>(input, output, dimension);
	STP_CHECK_CUDA(cudaGetLastError());
}

/* --------- Kernel Definition ----------- */

#include <device_launch_parameters.h>

using glm::vec2;

__global__ void curandInitKERNEL(STPHeightfieldKernel::STPcurand_t* const rng, const STPSeed_t seed, const unsigned int count) {
	//current working index
	const auto [index] = STPDeviceLaunchSetup::calcThreadIndex<1u>();
	if (index >= count) {
		return;
	}

	//the same seed but we are looking for different sequence
	curand_init(seed, static_cast<unsigned long long>(index), 0, &rng[index]);
}

//It's raining
#include <SuperTerrain+/GPGPU/STPRainDrop.cuh>

__global__ void hydraulicErosionKERNEL(STPHeightFloat_t* const heightmap_storage,
	const STPEnvironment::STPRainDropSetting* const raindrop_setting, const STPNearestNeighbourInformation nn_info,
	const STPErosionBrush brush, STPHeightfieldKernel::STPcurand_t* const rng) {
	//current working index
	const auto [index] = STPDeviceLaunchSetup::calcThreadIndex<1u>();
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

#include <cuda/std/limits>

__global__ void formatHeightmapKERNEL(STPHeightFloat_t* const input, STPHeightFixed_t* const output, const uvec2 dimension) {
	//the current working pixel
	const auto [x, y] = STPDeviceLaunchSetup::calcThreadIndex<2u>();
	const unsigned int index = x + y * dimension.x;
	//range check
	if (x >= dimension.x || y >= dimension.y) {
		return;
	}

	output[index] = static_cast<STPHeightFixed_t>(rintf(__saturatef(input[index]) * cuda::std::numeric_limits<STPHeightFixed_t>::max()));
}