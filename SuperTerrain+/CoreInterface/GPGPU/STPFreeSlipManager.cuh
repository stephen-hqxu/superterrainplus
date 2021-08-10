#pragma once
#ifndef _STP_FREESLIP_MANAGER_CUH_
#define _STP_FREESLIP_MANAGER_CUH_

#include <STPCoreDefine.h>
//CUDA
#include <cuda_runtime.h>
//Generator
#include "STPFreeSlipData.hpp"

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
		 * @brief STPFreeSlipManager is a utility wrapper on STPFreeSlipData
		 * such that it can make use of the index table and other data to "free-slip" index access on texture
		*/
		class STP_API STPFreeSlipManager {
		private:
			
			friend class STPFreeSlipGenerator;

			//A matrix of texture, it should be arranged in row major order.
			//The number of texture should be equal to the product or x and y defiend in FreeSlipRange
			//The size of the texture should be equal to FreeSlipRange.x * FreeSlipRange.y * Dimension.x * Dimension.y * sizeof(float)
			float* Texture;

			/**
			 * @brief Init the free slip manager.
			 * @param texture The texture array of any map (heightmap biomemap etc.), all chunks should be arranged in a linear array
			 * @param data Contains all data required for free-slip indexing
			*/
			__host__ STPFreeSlipManager(float*, const STPFreeSlipData*);

		public:

			//All free-slip data
			const STPFreeSlipData Data;

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