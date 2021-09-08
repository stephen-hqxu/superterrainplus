#pragma once
#ifndef _STP_FREESLIP_MANAGER_CUH_
#define _STP_FREESLIP_MANAGER_CUH_

//CUDA
#include <cuda_runtime.h>
//Generator
#include "../../World/Diversity/STPBiomeDefine.h"
#include "STPFreeSlipData.hpp"

//Use device code only in .cu files and host code only in .cpp to avoid nvcc generating unnecessary host on device applications
#ifdef __CUDACC__
#define MANAGER_HOST_DEVICE_SWITCH __device__
#else
#define MANAGER_HOST_DEVICE_SWITCH __host__
#endif//__CUDACC__

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
		 * such that it can make use of the index table and other data to "free-slip" index access on texture.
		 * @tparam T The data type of the texture
		*/
		template<typename T>
		class STPFreeSlipManager {
		private:

			//A matrix of texture, it should be arranged in row major order.
			//The number of texture should be equal to the product or x and y defiend in FreeSlipRange
			//The size of the texture should be equal to FreeSlipRange.x * FreeSlipRange.y * Dimension.x * Dimension.y * sizeof(float)
			T* const Texture;

		public:

			//All free-slip data
			const STPFreeSlipData* const Data;

			/**
			 * @brief Init the free slip manager.
			 * @param texture The texture array of any map (heightmap biomemap etc.), all chunks should be arranged in a linear array
			 * @param data A pointer to all data required for free-slip indexing.
			 * The life-time of the data should be guaranteed.
			 * Depends on the usage, the pointer to data should be either on host or device side
			*/
			__host__ STPFreeSlipManager(T*, const STPFreeSlipData*);

			__host__ ~STPFreeSlipManager();

			/**
			 * @brief Convert global index to local index and return the reference value.
			 * @param global Global index
			 * @return The pointer to the map pointed by the global index
			*/
			MANAGER_HOST_DEVICE_SWITCH T& operator[](unsigned int);

			/**
			 * @brief Convert global index to local index and return the const reference value
			 * @param global Global index
			 * @return Constant reference to the map pointed by the global index
			*/
			MANAGER_HOST_DEVICE_SWITCH const T& operator[](unsigned int) const;

			/**
			 * @brief Convert global index to local index
			 * @param global Global index
			 * @return Local index
			*/
			MANAGER_HOST_DEVICE_SWITCH unsigned int operator()(unsigned int) const;

		};

		typedef STPFreeSlipManager<float> STPFreeSlipFloatManager;
		typedef STPFreeSlipManager<STPDiversity::Sample> STPFreeSlipSampleManager;
		typedef STPFreeSlipManager<unsigned short> STPFreeSlipRenderManager;

	}
}
//We need to inline definitions, not because of template, but device function cannot be exported as shared library
#include "STPFreeSlipManager.inl"
#endif//_STP_FREESLIP_MANAGER_CUH_