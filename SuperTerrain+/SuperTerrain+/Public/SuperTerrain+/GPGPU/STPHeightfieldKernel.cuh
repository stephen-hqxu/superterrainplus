#pragma once
//Please include this header in source file (.cu) only
#ifndef STP_IMPLEMENTATION
#error __FILE__ is an internal CUDA kernel wrapper and is not intended to be called from external environment
#endif//STP_IMPLEMENTATION

#ifndef _STP_HEIGHTFIELD_KERNEL_CUH_
#define _STP_HEIGHTFIELD_KERNEL_CUH_

#include "../World/STPWorldMapPixelFormat.hpp"
//Memory
#include "../Utility/Memory/STPSmartDeviceMemory.h"
//Nearest-Neighbour
#include "../World/Chunk/STPNearestNeighbourInformation.hpp"
#include "../World/Chunk/STPErosionBrush.hpp"
//Settings
#include "../Environment/STPHeightfieldSetting.h"

//CUDA
#include <curand_kernel.h>
#include <cuda_runtime.h>

//GLM
#include <glm/vec2.hpp>

namespace SuperTerrainPlus {

	/**
	 * @brief STPHeightfieldKernel is a wrapper to CUDA kernel for heightfield generation related global functions.
	 * It provides smart kernel launch size calculators for all kernel calls.
	*/
	namespace STPHeightfieldKernel {

		typedef curandStatePhilox4_32_10 STPcurand_t;
		typedef STPSmartDeviceMemory::STPDevice<STPcurand_t[]> STPcurand_arr;
		/**
		 * @brief Init the curand generator for each thread
		 * @param seed The seed for each generator
		 * @param count The number of rng to be initialised.
		 * @param stream The CUDA stream.
		 * @return A random number generator array on device memory with managed smart pointer. Array is allocated with the number of count specified.
		*/
		__host__ STPcurand_arr curandInit(STPSeed_t, unsigned int, cudaStream_t);

		/**
		 * @brief Performing hydraulic erosion for the given heightmap terrain.
		 * @param height_storage The floating point heightmap with all neighbours merged together.
		 * Heightmap must be available in device memory.
		 * @param raindrop_settings The settings to use to erosion the heightmap, must be in device memory space.
		 * @param nn_info The information about the nearest neighbour of chunk, allowing *free-slip* erosion.
		 * @param brush The information about the generated erosion brush.
		 * @param raindrop_count The number of raindrop to spawn and erode the terrain.
		 * @param rng The random number generator map sequence, independent for each rain drop.
		 * @param stream Specify a CUDA stream work will be submitted to.
		*/
		__host__ void hydraulicErosion(STPHeightFloat_t*, const STPEnvironment::STPRainDropSetting*, const STPNearestNeighbourInformation&,
			const STPErosionBrush&, unsigned int, STPcurand_t*, cudaStream_t);

		/**
		 * @brief Texture channel format conversion. Floating point pixel channel to fixed point.
		 * @param input The input floating-point texture. Must be in the range of [0.0f, 1.0f].
		 * @param output The output fixed-point texture.
		 * @param dimension The dimension (number of pixel) of the texture.
		 * @param stream Specify a CUDA stream work will be submitted to
		*/
		__host__ void formatHeightmap(STPHeightFloat_t*, STPHeightFixed_t*, glm::uvec2, cudaStream_t);

	}

}
#endif//_STP_HEIGHTFIELD_KERNEL_CUH_