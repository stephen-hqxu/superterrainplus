#pragma once
#ifndef _STP_FREESLIP_GENERATOR_CUH_
#define _STP_FREESLIP_GENERATOR_CUH_

#include <STPCoreDefine.h>
//System
#include <memory>
#include <optional>
//Data Structure
#include <vector>
#include <tuple>
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
			 * @brief STPFreeSlipLocation denotes where tje free-slip data will be available.
			 * Once retrieved, the data retrieved can only be used in designated memory space
			*/
			enum class STPFreeSlipLocation : unsigned char {
				HostMemory = 0x00u,
				DeviceMemory = 0x01u
			};

			/**
			 * @brief STPFreeSlipManagerAdaptor is a top-level wrapper to STPFreeSlipManager.
			 * It stores information about free-slip and the texture, adapts and generates STPFreeSlipManager depends on usage.
			 * Note that life-time of the texture stored in returned STPFreeSlipManager is controlled by the calling STPFreeSlipManagerAdaptor.
			 * When the bounded STPFreeSlipManagerAdaptor is destroyed, free-slip texture can be optionally copied back to the original buffer.
			 * @tparam T The data type of the texture
			*/
			template<typename T>
			class STP_API STPFreeSlipManagerAdaptor {
			public:

				//reference to pointers to texture in a free-slip range in row-major order
				typedef std::vector<T*> STPFreeSlipTexture;

			private:

				friend class STPFreeSlipGenerator;

				//The generator that the free-slip manager adaptor is bounded to
				const STPFreeSlipGenerator& Generator;

				//The free-slip texture buffer stored separately
				STPFreeSlipTexture& Buffer;
				//CUDA stream for device memory allocation and copy.
				//Provide 0 to use default stream
				cudaStream_t Stream;

				//Allocated pinned memory
				mutable T* PinnedMemoryBuffer;
				//The previously integrated texture and its location
				//0: Set to false to copy the merged texture buffer back to the original array
				//1: The number of channel per pixel
				//2: Pointer to merged free-slip texture.
				//3: The location of the pointer
				mutable std::optional<std::tuple<T*, unsigned char, STPFreeSlipLocation, bool>> Integration;

				/**
				 * @brief Init the free-slip manager adaptor
				 * @param texture The ranged of free-slip texture per chunk which will be used to indexed in a free-slip manner.
				 * @param generator The free-slip generator that the adaptor will be bounded to.
				 * It will export data to free-slip manager based on the generator chosen
				 * @param stream CUDA stream for async device memory allocation and copy
				*/
				__host__ STPFreeSlipManagerAdaptor(STPFreeSlipTexture&, const STPFreeSlipGenerator&, cudaStream_t);

			public:

				__host__ ~STPFreeSlipManagerAdaptor();

				/**
				 * @brief Get the free-slip manager adaptor can retrieve different types of free-slip manager based on usage.
				 * @param location The memory space where the free-slip manager will be used.
				 * @param read_only Decide if the texture pointers provided should be read-only.
				 * If it's set to false, when the destructor of the adaptor is called, merged free-slip texture will be disintegrated and copied back to the original array of buffers.
				 * @param channel Denote the number of channel per pixel.
				 * @return The free-slip manager.
				 * Note that the manager is bounded to the current generator, meaning all underlying contents will be managed and become invalid once generator is deleted.
				 * The underlying merged texture pointer is managed by the current adaptor and will be freed when the adaptor got destroyed
				*/
				__host__ STPFreeSlipManager<T> operator()(STPFreeSlipLocation, bool, unsigned char) const;

			};

			typedef STPFreeSlipManagerAdaptor<float> STPFreeSlipFloatManagerAdaptor;
			typedef STPFreeSlipManagerAdaptor<STPDiversity::Sample> STPFreeSlipSampleManagerAdaptor;
			typedef STPFreeSlipManagerAdaptor<unsigned short> STPFreeSlipRenderManagerAdaptor;

		private:

			//Make a copy of global-local index table on host side
			std::unique_ptr<unsigned int[]> Index_Host;
			//Same for device side
			unsigned int* Index_Device;
			//Freeslip data copy on device side, the device index table is contained
			STPFreeSlipData* Data_Device;

			//Memory pool
			//TODO: no memory pool for pinned memory yet
			cudaMemPool_t DevicePool;

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
			 * @brief Set CUDA device memory pool
			 * @param device_mempool The CUDA memory pool.
			*/
			__host__ void setDeviceMemPool(cudaMemPool_t);

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
			 * @tparam T The type of texture
			 * @param texture The ranged of free-slip texture per chunk which will be used to indexed in a free-slip manner.
			 * @param stream CUDA stream for async device memory allocation and copy
			 * @return The free-slip manager adaptor which will be bounded to the current generator
			*/
			template<typename T>
			__host__ STPFreeSlipManagerAdaptor<T> getAdaptor(typename STPFreeSlipManagerAdaptor<T>::STPFreeSlipTexture&, cudaStream_t) const;

		};

	}
}
#endif//_STP_FREESLIP_GENERATOR_CUH_