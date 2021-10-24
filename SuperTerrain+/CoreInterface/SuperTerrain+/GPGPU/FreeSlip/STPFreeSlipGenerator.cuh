#pragma once
#ifndef _STP_FREESLIP_GENERATOR_CUH_
#define _STP_FREESLIP_GENERATOR_CUH_

#include <SuperTerrain+/STPCoreDefine.h>
//System
#include <memory>
//CUDA
#include <cuda_runtime.h>
//Free-slip Data
#include "STPFreeSlipLocation.hpp"
#include "STPFreeSlipTextureBuffer.h"
#include "STPFreeSlipManager.cuh"
#include "../../Utility/STPSmartDeviceMemory.h"

namespace SuperTerrainPlus::STPCompute {

	/**
	 * @brief STPFreeSlipGenerator provides a center chunk for erosion and some neighbour chunks that hold data access out of the center chunk.
	 * It will the convert global index to local index, such that indeices can "free slip" out of the center chunk.
	*/
	class STP_API STPFreeSlipGenerator {
	public:

		/**
		 * @brief STPFreeSlipManagerAdaptor is a top-level wrapper to STPFreeSlipManager.
		 * STPFreeSlipManagerAdaptor is a very light-weight class and can be created, copied and moved cheaply
		 * @tparam T The data type of the texture
		*/
		template<typename T>
		class STP_API STPFreeSlipManagerAdaptor {
		private:

			friend class STPFreeSlipGenerator;

			//The generator that the free-slip manager adaptor is bounded to
			const STPFreeSlipGenerator& Generator;
			//Free-slip texture
			STPFreeSlipTextureBuffer<T>& Texture;

			/**
			 * @brief Init the free-slip manager adaptor
			 * @param buffer The ranged of free-slip texture per chunk which will be used to indexed in a free-slip manner.
			 * @param generator The free-slip generator that the adaptor will be bounded to.
			 * It will export data to free-slip manager based on the generator chosen
			*/
			__host__ STPFreeSlipManagerAdaptor(STPFreeSlipTextureBuffer<T>&, const STPFreeSlipGenerator&);

		public:

			__host__ ~STPFreeSlipManagerAdaptor();

			/**
			 * @brief Get the free-slip manager adaptor can retrieve different types of free-slip manager based on usage.
			 * Repetitive call to get multiple STPFreeSlipManager is cheap, but should read the manual in STPFreeSlipTextureBuffer for the behaviour.
			 * @param location The memory space where the free-slip manager will be used.
			 * The first call will be safe.
			 * The behaviour is undefined if the subsequent location is more strict than the location used in first-time call.
			 * (strictness (relaxed to strict): host -> device)
			 * @return The free-slip manager.
			 * Note that the manager is bounded to the generator and texture buffer,
			 * meaning all underlying contents will be managed and become invalid once they are deleted.
			 * @see STPFreeSlipTextureBuffer to read the effect of requesting multiple STPFreeSlipManagers from the same adaptor.
			*/
			__host__ STPFreeSlipManager<T> operator()(STPFreeSlipLocation) const;

		};

		typedef STPFreeSlipManagerAdaptor<float> STPFreeSlipFloatManagerAdaptor;
		typedef STPFreeSlipManagerAdaptor<STPDiversity::Sample> STPFreeSlipSampleManagerAdaptor;
		typedef STPFreeSlipManagerAdaptor<unsigned short> STPFreeSlipRenderManagerAdaptor;

	private:

		STPFreeSlipData Data;
		//Make a copy of global-local index table on host side
		std::unique_ptr<unsigned int[]> Index_Host;
		//Same for device side
		STPSmartDeviceMemory::STPDeviceMemory<unsigned int[]> Index_Device;
		//Freeslip data copy on device side, the device index table is contained
		STPSmartDeviceMemory::STPDeviceMemory<STPFreeSlipData> Data_Device;

		/**
		 * @brief Initialise the local global index lookup table
		*/
		__host__ void initLocalGlobalIndexCUDA();

	public:

		/**
		 * @brief Init STPFreeSlipGenerator and generate global-local index table.
		 * No reference is retained after the function returned
		 * @param range Free slip range in the unit of chunk
		 * @param mapSize The size of the each heightmap
		*/
		__host__ STPFreeSlipGenerator(glm::uvec2, glm::uvec2);

		__host__ ~STPFreeSlipGenerator();

		__host__ STPFreeSlipGenerator(const STPFreeSlipGenerator&) = delete;

		__host__ STPFreeSlipGenerator(STPFreeSlipGenerator&&) = delete;

		__host__ STPFreeSlipGenerator& operator=(const STPFreeSlipGenerator&) = delete;

		__host__ STPFreeSlipGenerator& operator=(STPFreeSlipGenerator&&) = delete;

		/**
		 * @brief Get the dimension of each texture.
		 * @return The pointer to dimension
		*/
		__host__ const glm::uvec2& getDimension() const;

		/**
		 * @brief Get the number of free-slip chunk
		 * @return The poiner to free-slip chunk
		*/
		__host__ const glm::uvec2& getFreeSlipChunk() const;

		/**
		 * @brief Get the number of pixel in total in free-slip logic
		 * @return The pointer to the free-slip range
		*/
		__host__ const glm::uvec2& getFreeSlipRange() const;

		/**
		 * @brief Get the free-slip manager adaptor, which will dynamically determine free-slip configuration to use from the generator based on chosen type.
		 * The obtained adaptor is bounded to the current generator.
		 * @tparam T The type of texture
		 * @param buffer The free-slip texture buffer will be used to create free-slip texture for different memory space.
		 * After binding the buffer to the returned adaptor, buffer will be managed by the adaptor automatically.
		 * However the life-time of the buffer should be guaranteed until the adaptor is no longer in used.
		 * @return The free-slip manager adaptor which will be bounded to the current generator
		*/
		template<typename T>
		__host__ STPFreeSlipManagerAdaptor<T> operator()(STPFreeSlipTextureBuffer<T>&) const;

	};

}
#endif//_STP_FREESLIP_GENERATOR_CUH_