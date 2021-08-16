#pragma once
#ifndef _STP_FREESLIP_GENERATOR_CUH_
#define _STP_FREESLIP_GENERATOR_CUH_

#include <STPCoreDefine.h>
//System
#include <memory>
//CUDA
#include <cuda_runtime.h>
//Free-slip Data
#include "STPFreeSlipManager.cuh"

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
		 * @brief STPFreeSlipGenerator provides a center chunk for erosion and some neighbour chunks that hold data access out of the center chunk.
		 * It will the convert global index to local index, such that indeices can "free slip" out of the center chunk.
		*/
		class STP_API STPFreeSlipGenerator : private STPFreeSlipData {
		public:

			/**
			 * @brief STPFreeSlipManagerAdaptor is a top-level wrapper to STPFreeSlipManager.
			 * It stores information about free-slip and the texture, and generate STPFreeSlipManager depends on usage.
			*/
			class STP_API STPFreeSlipManagerAdaptor {
			public:

				/**
				 * @brief STPFreeSlipManagerType denotes the type of free-slip manager to retrieve.
				 * Once retrieved, the manager can only be used in designated memory space
				*/
				enum class STP_API STPFreeSlipManagerType : unsigned char {
					HostManager = 0x00u,
					DeviceManager = 0x01u
				};

			private:

				friend class STPFreeSlipGenerator;

				//The generator that the free-slip manager adaptor is bounded to
				const STPFreeSlipGenerator& Generator;
				//The stored pointer to texture
				void* const Texture;

				/**
				 * @brief Init the free-slip manager adaptor
				 * @param texture The texture which will be used to indexed in a free-slip manner
				 * @param generator The free-slip generator that the adaptor will be bounded to.
				 * It will export data to free-slip manager based on the generator chosen
				*/
				__host__ STPFreeSlipManagerAdaptor(void*, const STPFreeSlipGenerator&);

				/**
				 * @brief Get the free-slip manager which deals with specific texture format
				 * @tparam M The format of the manager
				 * @param type The memory space where the manager will reside
				 * @return The manager which deals with that type of texture
				*/
				template<typename T>
				__host__ STPFreeSlipManager<T> getTypedManager(STPFreeSlipManagerType) const;

			public:

				__host__ ~STPFreeSlipManagerAdaptor();

				/**
				 * @brief Get the free-slip manager adaptor can retrieve different types of free-slip manager based on usage.
				 * @tparam M The data type on the free-slip manager, so it will convert texture to that data type.
				 * @param type The memory space where the free-slip manager will be used.
				 * @return The free-slip manager.
				 * Note that the manager is bounded to the current generator, meaning all underlying contents will be managed and become invalid once generator is deleted.
				*/
				template<typename M>
				__host__ STPFreeSlipManager<M> getManager(STPFreeSlipManagerType) const;

			};

		private:

			//Make a copy of global-local index table on host side
			std::unique_ptr<unsigned int[]> Index_Host;
			//Same for device side
			unsigned int* Index_Device;
			//Freeslip data copy on device side, the device index table is contained
			STPFreeSlipData* Data_Device;

			/**
			 * @brief Initialise the local global index lookup table
			*/
			__host__ void initLocalGlobalIndexCUDA();

			/**
			 * @brief Try to delete the device index table (if exists)
			*/
			__host__ void clearDeviceIndex() noexcept;

		public:

			/**
			 * @brief Init STPFreeSlipGenerator and generate global-local index table
			 * @param range Free slip range in the unit of chunk
			 * @param mapSize The size of the each heightmap
			*/
			__host__ STPFreeSlipGenerator(uint2, uint2);

			__host__ ~STPFreeSlipGenerator();

			__host__ STPFreeSlipGenerator(const STPFreeSlipGenerator&) = delete;

			__host__ STPFreeSlipGenerator(STPFreeSlipGenerator&&) = delete;

			__host__ STPFreeSlipGenerator& operator=(const STPFreeSlipGenerator&) = delete;

			__host__ STPFreeSlipGenerator& operator=(STPFreeSlipGenerator&&) = delete;

			/**
			 * @brief Get the dimension of each texture.
			 * @return The pointer to dimension
			*/
			__host__ const uint2& getDimension() const;

			/**
			 * @brief Get the number of free-slip chunk
			 * @return The poiner to free-slip chunk
			*/
			__host__ const uint2& getFreeSlipChunk() const;

			/**
			 * @brief Get the number of pixel in total in free-slip logic
			 * @return The pointer to the free-slip range
			*/
			__host__ const uint2& getFreeSlipRange() const;

			/**
			 * @brief Get the free-slip manager adaptor, which will dynamically determine free-slip configuration to use from the generator based on chosen type.
			 * The obtained adaptor is bounded to the current generator.
			 * @param texture The texture that will be bounded to the 
			 * @return The free-slip manager adaptor which will be bounded to the current generator
			*/
			__host__ STPFreeSlipManagerAdaptor operator()(void*) const;

		};

	}
}
#endif//_STP_FREESLIP_GENERATOR_CUH_