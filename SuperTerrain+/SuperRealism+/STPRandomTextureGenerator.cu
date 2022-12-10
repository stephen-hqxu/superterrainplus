#include <SuperRealism+/Utility/STPRandomTextureGenerator.cuh>

#include <curand_kernel.h>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>
#include <SuperTerrain+/Exception/STPNumericDomainError.h>

#include <SuperTerrain+/Utility/Memory/STPSmartDeviceObject.h>

//GLM
#include <glm/vec2.hpp>

using glm::uvec2;
using glm::uvec3;

using namespace SuperTerrainPlus::STPRealism;

/* ------------------------ Kernel declaration ------------------------------- */

/* TODO:
This seems to be a bug in CUDA that, for a template global function,
	both declaration and definition must have corresponding arguments with `const`.
According to C++ specification this should not mandated.
Bug report: https://forums.developer.nvidia.com/t/invaliddevicefunction-error-when-launching-templated-global-function/234870?u=stephen.hqxu
*/

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
__global__ static void generateRandomTextureKERNEL(const cudaSurfaceObject_t, const uvec3, const unsigned long long, const T, const T);

template<typename T>
__host__ void STPRandomTextureGenerator::generate(const cudaArray_t output, const uvec3 dimension,
	const unsigned long long seed, const T min, const T max) {
	//range check
	STP_ASSERTION_NUMERIC_DOMAIN(dimension.x > 0u && dimension.y > 0u && dimension.z > 0u, "The dimension of the random texture must be all positive");

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

/**
 * @brief Generate a random number.
 * Force no inline to prevent regenerating huge CURAND code for different template instantiation.
 * @param seed The seed.
 * @param index The index of the kernel thread.
 * @return A normalised random number.
*/
__device__ __noinline__ static float getRandomNumber(const unsigned long long seed, const unsigned int index) {
	curandStatePhilox4_32_10_t state;
	curand_init(seed, static_cast<unsigned long long>(index), 0ull, &state);
	return curand_uniform(&state);
}

template<typename T>
__global__ void generateRandomTextureKERNEL(const cudaSurfaceObject_t surface, const uvec3 dimension,
	const unsigned long long seed, const T base, const T range) {
	//get current invocation
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x,
		y = (blockIdx.y * blockDim.y) + threadIdx.y,
		z = (blockIdx.z * blockDim.z) + threadIdx.z;
	//sanity check
	if (x >= dimension.x || y >= dimension.y || z >= dimension.z) {
		return;
	}
	const unsigned int index = x + dimension.x * y + (dimension.x * dimension.y) * z;

	//generate and scale the random number
	const T rand = static_cast<T>(rintf(getRandomNumber(seed, index) * range + base));

	//output
	surf3Dwrite(rand, surface, x * sizeof(T), y, z, cudaBoundaryModeTrap);
}

//Template instantiation
#define GENERATE(TYPE) \
template __host__ void STPRandomTextureGenerator::generate<TYPE>(cudaArray_t, uvec3, unsigned long long, TYPE, TYPE)

GENERATE(unsigned char);