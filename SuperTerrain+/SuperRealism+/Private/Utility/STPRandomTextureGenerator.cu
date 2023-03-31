#include <SuperRealism+/Utility/STPRandomTextureGenerator.cuh>

//Error
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>
#include <SuperTerrain+/Exception/STPNumericDomainError.h>
//Utility
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceObject.h>
#include <SuperTerrain+/Utility/STPDeviceLaunchSetup.cuh>

//GLM
#include <glm/vec2.hpp>

//GLAD
#include <glad/glad.h>
//CUDA
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

#include <tuple>
#include <algorithm>

using glm::uvec2;
using glm::uvec3;

using std::tuple, std::make_tuple;

namespace DevObj = SuperTerrainPlus::STPSmartDeviceObject;
using namespace SuperTerrainPlus::STPRealism;

using SuperTerrainPlus::STPSeed_t;
using DevObj::STPGraphicsResource, DevObj::STPSurface;
using DevObj::makeGLImageResource, DevObj::makeSurface;

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
__global__ static void generateRandomTextureKERNEL(const cudaSurfaceObject_t, const uvec3, const STPSeed_t, const T, const T);

/**
 * @brief Prepare the GL texture by registering it with CUDA.
 * @param random_image The GL texture object where the generated random image will be stored.
 * @return The registered graphics resource, the surface constructed from such graphics resource and the dimension of the image.
*/
__host__ static tuple<STPGraphicsResource, STPSurface, uvec3> prepareGLRandomImage(const STPTexture& random_image, const cudaStream_t stream) {
	//register buffer
	STPGraphicsResource res_managed = makeGLImageResource(*random_image, random_image.target(), cudaGraphicsRegisterFlagsWriteDiscard);
	cudaGraphicsResource_t res = res_managed.get();

	//create surface
	cudaArray_t random_buffer;
	STP_CHECK_CUDA(cudaGraphicsMapResources(1, &res, stream));
	STP_CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&random_buffer, res, 0u, 0u));

	cudaResourceDesc desc = {};
	desc.resType = cudaResourceTypeArray;
	desc.res.array.array = random_buffer;

	//get dimension, a texture is always an array memory, so the buffer size is specified in element count rather than byte
	constexpr static auto toDimension = [](const size_t extent) constexpr noexcept -> unsigned int {
		return std::max(static_cast<unsigned int>(extent), 1u);
	};
	cudaExtent buffer_dim;
	STP_CHECK_CUDA(cudaArrayGetInfo(nullptr, &buffer_dim, nullptr, random_buffer));
	const auto [width, height, depth] = buffer_dim;

	return make_tuple(
		std::move(res_managed),
		makeSurface(desc),
		uvec3(
			toDimension(width),
			toDimension(height),
			toDimension(depth)
		)
	);
}

template<typename T>
__host__ void STPRandomTextureGenerator::generate(const STPTexture& output, const STPSeed_t seed,
	const T min, const T max, const cudaStream_t stream) {
	const auto [managed_resource, managed_surface, dimension] = prepareGLRandomImage(output, stream);
	//calculate launch configuration, always treat this as a 3D texture
	const auto [gridSize, blockSize] =
		STPDeviceLaunchSetup::determineLaunchConfiguration<3u>(generateRandomTextureKERNEL<T>, dimension);

	const T base = min, range = (max - min);
	generateRandomTextureKERNEL<<<gridSize, blockSize, 0u, stream>>>(managed_surface.get(), dimension, seed, base, range);
	STP_CHECK_CUDA(cudaGetLastError());

	//clean up
	cudaGraphicsResource_t res = managed_resource.get();
	STP_CHECK_CUDA(cudaGraphicsUnmapResources(1, &res, stream));
}

/* ---------------------------- Kernel definition ------------------------------------ */

//CUDA
#include <curand_kernel.h>

/**
 * @brief Generate a random number.
 * Force no inline to prevent regenerating huge CURAND code for different template instantiation.
 * @param seed The seed.
 * @param index The index of the kernel thread.
 * @return A normalised random number.
*/
__device__ static float getRandomNumber(const STPSeed_t seed, const unsigned int index) {
	curandStatePhilox4_32_10_t state;
	curand_init(seed, static_cast<unsigned long long>(index), 0ull, &state);
	return curand_uniform(&state);
}

template<typename T>
__global__ void generateRandomTextureKERNEL(const cudaSurfaceObject_t surface, const uvec3 dimension,
	const STPSeed_t seed, const T base, const T range) {
	//get current invocation
	const auto [x, y, z] = SuperTerrainPlus::STPDeviceLaunchSetup::calcThreadIndex<3u>();
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
template __host__ void STPRandomTextureGenerator::generate<TYPE>(const STPTexture&, STPSeed_t, TYPE, TYPE, cudaStream_t)

GENERATE(unsigned char);