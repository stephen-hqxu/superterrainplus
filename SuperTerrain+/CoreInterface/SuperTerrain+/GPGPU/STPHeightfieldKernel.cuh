#pragma once
//Please include this header in source file (.cu) only
#ifndef STP_IMPLEMENTATION
#error __FILE__ is an internal CUDA kernel wrapper and is not intended to be called from external environment
#endif//STP_IMPLEMENTATION

#ifndef _STP_HEIGHTFIELD_KERNEL_CUH_
#define _STP_HEIGHTFIELD_KERNEL_CUH_

//CUDA
#include <curand_kernel.h>
#include <cuda_runtime.h>
//Memory
#include "../Utility/STPSmartDeviceMemory.h"

//GLM
#include <glm/vec2.hpp>

//FreeSlip
#include "../World/Chunk/FreeSlip/STPFreeSlipManager.cuh"
//Settings
#include "../Environment/STPHeightfieldSetting.h"

namespace SuperTerrainPlus::STPCompute {

	/**
	 * @brief STPHeightfieldKernel is a wrapper to CUDA kernel for heightfield generation related global functions.
	 * It provides smart kernel launch size calculators for all kernel calls.
	*/
	namespace STPHeightfieldKernel {

		typedef curandStatePhilox4_32_10 STPcurand_t;
		typedef STPSmartDeviceMemory::STPDeviceMemory<STPcurand_t[]> STPcurand_arr;
		/**
		 * @brief Init the curand generator for each thread
		 * @param seed The seed for each generator
		 * @param count The number of rng to be initialised.
		 * @return A random number generator array on device memory with managed smart pointer. Array is allocated with the number of count specified.
		*/
		__host__ STPcurand_arr curandInit(unsigned long long, unsigned int);

		typedef STPSmartDeviceMemory::STPDeviceMemory<unsigned int[]> STPIndexTable;
		/**
		 * @brief Generate a new global to local index table
		 * @param chunkRange The number of chunk (or locals)
		 * @param tableSize The x,y dimension of the table
		 * @param mapSize The dimension of the map
		 * @return The generated table. Allocated and managed by smart pointer with
		 * size sizeof(unsigned int) * chunkRange.x * mapSize.x * chunkRange.y * mapSize.y.
		*/
		__host__ STPIndexTable initGlobalLocalIndex(glm::uvec2, glm::uvec2, glm::uvec2);

		/**
		 * @brief Performing hydraulic erosion for the given heightmap terrain
		 * @param height_storage The floating point heightmap with global-local free-slip management.
		 * Heightmap must be available in device memory.
		 * @param heightfield_settings The settings to use to generate heightmap, must be in device memory space
		 * @param brush_size The number of erosion brush. The brush weight must have the same size as the brush indices
		 * @param raindrop_count The number of raindrop to spawn and erode the terrain
		 * @param rng The random number generator map sequence, independent for each rain drop
		 * @param stream Specify a CUDA stream work will be submitted to
		*/
		__host__ void hydraulicErosion(STPFreeSlipFloatManager, const STPEnvironment::STPHeightfieldSetting*, 
			unsigned int, unsigned int, STPcurand_t*, cudaStream_t);

		/**
		 * @brief Texture channel format convertion. FP32 to INT16.
		 * Perform the following operation: output = normalise(index) * INT16_MAX
		 * @param input The input FP32 texture. Must be in the range of [0.0f, 1.0f].
		 * @param output The ouput INT16 texture.
		 * @param dimension The dimension (number of pixel) of the texture.
		 * @param channel The number of channel per pixel
		 * @param stream Specify a CUDA stream work will be submitted to
		*/
		__host__ void texture32Fto16(float*, unsigned short*, glm::uvec2, unsigned int, cudaStream_t);

	}

}
#endif//_STP_HEIGHTFIELD_KERNEL_CUH_