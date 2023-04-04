#include <SuperTerrain+/GPGPU/STPHeightfieldKernel.cuh>

#include <SuperTerrain+/Utility/STPDeviceLaunchSetup.cuh>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

/* --------- Kernel Declaration ----------- */

using namespace SuperTerrainPlus;

#include <glm/geometric.hpp>

using glm::uvec2;

__global__ static void initialiseRainDropGeneratorKERNEL(STPHeightfieldKernel::STPRainDropGenerator*, const STPEnvironment::STPRainDropSetting*);

__global__ static void hydraulicErosionKERNEL(STPHeightFloat_t*, const STPEnvironment::STPRainDropSetting*,
	STPHeightfieldKernel::STPRainDropGenerator*, STPNearestNeighbourInformation, STPErosionBrush);

__global__ static void formatHeightmapKERNEL(STPHeightFloat_t*, STPHeightFixed_t*, uvec2);

__host__ STPHeightfieldKernel::STPRainDropGeneratorMemory STPHeightfieldKernel::initialiseRainDropGenerator(
	const STPEnvironment::STPRainDropSetting* const raindrop_setting, const unsigned int count, const cudaStream_t stream) {
	const auto [gridSize, blockSize] = STPDeviceLaunchSetup::determineLaunchConfiguration<1u>(initialiseRainDropGeneratorKERNEL, count);

	//allocating spaces for rng storage array
	//because RNG storage is persistent, i.e., we will be keep reusing it once allocated, no need to allocate from a memory pool.
	STPRainDropGeneratorMemory rng = STPSmartDeviceMemory::makeDevice<STPRainDropGenerator[]>(count);
	//and send to kernel to init rng sequences
	initialiseRainDropGeneratorKERNEL<<<gridSize, blockSize, 0, stream>>>(rng.get(), raindrop_setting);
	STP_CHECK_CUDA(cudaGetLastError());

	return rng;
}

__host__ void STPHeightfieldKernel::hydraulicErosion(STPHeightFloat_t* const heightmap_storage,
	const STPEnvironment::STPRainDropSetting* const raindrop_setting, STPRainDropGenerator* const raindrop_gen,
	const unsigned int raindrop_count, const STPNearestNeighbourInformation& nn_info, const STPErosionBrush& brush,
	const cudaStream_t stream) {
	//brush contains two components: weights (float) and indices (int)
	const unsigned int erosionBrushCache_size = brush.BrushSize * (sizeof(int) + sizeof(float));
	const auto [gridSize, blockSize] = STPDeviceLaunchSetup::determineLaunchConfiguration<1u>(hydraulicErosionKERNEL,
		raindrop_count, erosionBrushCache_size);

	hydraulicErosionKERNEL<<<gridSize, blockSize, erosionBrushCache_size, stream>>>(
		heightmap_storage, raindrop_setting, raindrop_gen, nn_info, brush);
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

__global__ void initialiseRainDropGeneratorKERNEL(STPHeightfieldKernel::STPRainDropGenerator* const rng,
	const STPEnvironment::STPRainDropSetting* const raindrop) {
	//current working index
	const auto [index] = STPDeviceLaunchSetup::calcThreadIndex<1u>();
	if (index >= raindrop->RainDropCount) {
		return;
	}

	//the same seed but we are looking for different sequence
	curand_init(raindrop->Seed, static_cast<unsigned long long>(index), 0, rng + index);
}

//It's raining
#include <SuperTerrain+/GPGPU/STPRainDrop.cuh>

__global__ void hydraulicErosionKERNEL(STPHeightFloat_t* const heightmap_storage,
	const STPEnvironment::STPRainDropSetting* const raindrop_setting,
	STPHeightfieldKernel::STPRainDropGenerator* const rng, const STPNearestNeighbourInformation nn_info,
	const STPErosionBrush brush) {
	//current working index
	const auto [index] = STPDeviceLaunchSetup::calcThreadIndex<1u>();
	if (index >= raindrop_setting->RainDropCount) {
		return;
	}

	//convert to (base, dimension - 1]
	//range: dimension
	//Generate the raindrop at the central chunk only
	const uvec2 base = uvec2(nn_info.MapSize - 1u),
		range = uvec2((nn_info.ChunkNearestNeighbour / 2u) * nn_info.MapSize);

	//generating random location
	//first we generate the number (0.0f, 1.0f]
	vec2 initPos;
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