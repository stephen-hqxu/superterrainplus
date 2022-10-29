#include <SuperRealism+/Utility/STPRandomTextureGenerator.cuh>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>
#include <SuperTerrain+/Exception/STPBadNumericRange.h>

#include <SuperTerrain+/Utility/Memory/STPSmartDeviceObject.h>

//GLM
#include <glm/vec2.hpp>

using glm::uvec2;
using glm::uvec3;

using namespace SuperTerrainPlus::STPRealism;

/* ------------------------ Kernel declaration ------------------------------- */

/**
 * @brief Call CUDA kernel to generate a random texture.
 * @tpara T The type of the texture.
 * @param surface The output surface object for writing the noise.
 * @param dimension The dimension of the texture.
 * @param seed The seed value for the random number generator.
 * @param base The starting number in the random range.
 * @param range The distance from the max and base.
*/
template<typename T>
__global__ static void generateRandomTextureKERNEL(cudaSurfaceObject_t, uvec3, unsigned long long, T, T);

template<typename T>
__host__ void STPRandomTextureGenerator::generate(cudaArray_t output, uvec3 dimension, unsigned long long seed, T min, T max) {
	//range check
	if (dimension.x == 0u || dimension.y == 0u || dimension.z == 0u) {
		throw STPException::STPBadNumericRange("Invalid dimension");
	}

	//calculate launch configuration
	int Mingridsize, blocksize;
	STP_CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&Mingridsize, &blocksize, &generateRandomTextureKERNEL<T>));

	//determine block size based on dimension
	uvec3 Dimblocksize;
	if (dimension.y == 1u && dimension.z == 1u) {
		//a 1D texture
		Dimblocksize = uvec3(static_cast<unsigned int>(blocksize), uvec2(1u));
	} else {
		const uvec2 Dimblocksize2D(32u, static_cast<unsigned int>(blocksize) / 32u);

		if (dimension.z == 1u) {
			//a 2D texture
			Dimblocksize = uvec3(Dimblocksize2D, 1u);
		} else {
			//a 3D texture
			Dimblocksize = uvec3(Dimblocksize2D.y, 8u, Dimblocksize2D.x / 8u);
		}
	}
	const uvec3 Dimgridsize = (dimension + Dimblocksize - 1u) / Dimblocksize;

	//prepare surface
	cudaResourceDesc desc = { };
	desc.resType = cudaResourceTypeArray;
	desc.res.array.array = output;

	STPSmartDeviceObject::STPSurface noise_buffer = STPSmartDeviceObject::makeSurface(desc);

	//computing
	const T base = min,
		range = (max - min);
	generateRandomTextureKERNEL<<<dim3(Dimgridsize.x, Dimgridsize.y, Dimgridsize.z),
		dim3(Dimblocksize.x, Dimblocksize.y, Dimblocksize.z)>>>(noise_buffer.get(), dimension, seed, base, range);
	STP_CHECK_CUDA(cudaDeviceSynchronize());
	STP_CHECK_CUDA(cudaGetLastError());
}

/* ---------------------------- Kernel definition ------------------------------------ */

//CUDA
#include <curand_kernel.h>
#include <device_launch_parameters.h>

//TODO: You can pick your RNG here.
typedef curandStateMRG32k3a_t STPTextureRNG;

template<typename T>
__global__ void generateRandomTextureKERNEL(cudaSurfaceObject_t surface, uvec3 dimension, unsigned long long seed, T base, T range) {
	//get current invocation
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y,
		z = (blockIdx.z * blockDim.z) + threadIdx.z;
	//sanity check
	if (x >= dimension.x || y >= dimension.y || z >= dimension.z) {
		return;
	}
	const unsigned int index = x + dimension.x * y + (dimension.x * dimension.y) * z;

	//init random number generator
	STPTextureRNG state;
	curand_init(seed, static_cast<unsigned long long>(index), 0ull, &state);
	//generate and scale the random number
	const T rand = static_cast<T>(curand_uniform(&state) * range + base);

	//output
	surf3Dwrite(rand, surface, x, y, z, cudaBoundaryModeTrap);
}

//Template instantiation
#define GENERATE(TYPE) \
template __host__ void STPRandomTextureGenerator::generate<TYPE>(cudaArray_t, uvec3, unsigned long long, TYPE, TYPE)

GENERATE(unsigned char);