#pragma once
#ifndef STP_IMPLEMENTATION
#error __FILE__ auto wraps index in layered row-major texture matrix from global to local but shall not be used in external environment
#endif//_STP_FREESLIP_MANAGER_CUH_

#ifndef _STP_FREESLIP_MANAGER_CUH_
#define _STP_FREESLIP_MANAGER_CUH_

//CUDA
#include <cuda_runtime.h>

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {
	/**
	 * @brief GPGPU compute suites for Super Terrain + program, powered by CUDA
	*/
	namespace STPCompute {

		/**
		 * @brief STPFreeSlipManager provides a center chunk for erosion and some neighbour chunks that hold data access out of the center chunk.
		 * It will the convert global index to local index, such that rain drop can "free slip" out of the center chunk.
		*/
		struct STPFreeSlipManager {
		private:

			friend class STPRainDrop;

			//A matrix of heightmap, it should be arranged in row major order.
			//The number of heightmap should be equal to the product or x and y defiend in FreeSlipRange
			//The size of the heightmap should be equal to FreeSlipRange.x * FreeSlipRange.y * Dimension.x * Dimension.y * sizeof(float)
			float* Heightmap;
			//A table that is responsible for conversion from global index to local index
			const unsigned int* Index;

		public:

			//The dimension of each map
			const uint2 Dimension;
			//The range of free slip in the unit of chunk
			const uint2 FreeSlipChunk;
			//number of element in a global row and column in the free slip range
			const uint2 FreeSlipRange;

			/**
			 * @brief Init the free slip manager.
			 * The center chunk will be determined automatically
			 * @param heightmap The heightmap array, all chunks should be arranged in a linear array
			 * @param index The lookup table to convert global index to local index.
			 * If nullptr is provided (meaning no lookup table), global index will be used directly to reference heightmap.
			 * @param range Free slip range in the unit of chunk
			 * @param mapSize The size of the each heightmap
			*/
			__host__ STPFreeSlipManager(float*, const unsigned int*, uint2, uint2);

			__host__ ~STPFreeSlipManager();

			/**
			 * @brief Convert global index to local index and return the reference value.
			 * @param global Global index
			 * @return The pointer to the map pointed by the global index
			*/
			__device__ float& operator[](unsigned int);

			/**
			 * @brief Convert global index to local index and return the const reference value
			 * @param global Global index
			 * @return Constant reference to the map pointed by the global index
			*/
			__device__ const float& operator[](unsigned int) const;

			/**
			 * @brief Convert global index to local index
			 * @param global Global index
			 * @return Local index
			*/
			__device__ unsigned int operator()(unsigned int) const;

		};

	}
}
#endif//_STP_FREESLIP_MANAGER_CUH_