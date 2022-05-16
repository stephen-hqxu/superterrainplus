#pragma once
#ifndef _STP_FREESLIP_TEXTURE_BUFFER_H_
#define _STP_FREESLIP_TEXTURE_BUFFER_H_

#include <SuperTerrain+/STPCoreDefine.h>
//System
#include <optional>
//Data Structure
#include <vector>
#include "../../Utility/Memory/STPMemoryPool.h"
//CUDA
#include <cuda_runtime.h>
//Free-Slip Data
#include "STPFreeSlipInformation.hpp"
//Memory
#include "../../Utility/Memory/STPSmartDeviceMemory.h"

#include "..//Diversity/STPBiomeDefine.h"

namespace SuperTerrainPlus {

	/**
	 * @brief STPFreeSlipTextureAttribute stores invariants for free-slip texture buffer management.
	 * It should be preserved by the caller so creation of STPFreeSlipTextureBuffer is light-weight.
	*/
	struct STPFreeSlipTextureAttribute {
	public:

		//Information about the free-slip texture
		STPFreeSlipInformation TextureInfo;

		//Memory Pool
		//Host memory pool is thread safe and can (and should!) be shared with other texture buffer objects
		mutable STPPinnedMemoryPool HostMemPool;
		cudaMemPool_t DeviceMemPool;

	};

	/**
	 * @brief STPFreeSlipTextureBuffer is a texture manager dedicated for free-slip neighbour system.
	 * It takes in a range of pointers to texture for each chunk, merge them into a large buffer aligned in row-major local-index.
	 * It stores information about free-slip and the texture, adapts and generates texture buffer depends on usage.
	 * Note that life-time of the pointer to the merged texture buffer is controlled by the calling STPFreeSlipTextureBuffer.
	 * When the bounded STPFreeSlipTextureBuffer is destroyed, free-slip texture can be optionally copied back to the original buffer.
	 * STPFreeSlipTextureBuffer is a light-weight class and can be created with ease
	 * @tparam T The data type of the texture
	*/
	template<typename T>
	class STP_API STPFreeSlipTextureBuffer {
	public:

		//reference to pointers to texture in a free-slip range in row-major order
		typedef std::vector<T*> STPFreeSlipTexture;

		/**
		 * @brief STPFreeSlipTextureData contains essential data for texture buffer
		*/
		struct STPFreeSlipTextureData {
		public:

			/**
			 * @brief STPMemoryMode contains set of rules for how the free-slip memory manager to react
			*/
			enum class STPMemoryMode : unsigned char {
				//Copy the initial buffer to merged texture
				//When the destructor of the adaptor is called, merged free-slip texture will be freed straight away
				ReadOnly = 0x00u,
				//No copy from the initial buffer, only allocation
				//When the destructor of the adaptor is called, merged free-slip texture will be disintegrated and copied back to the original array of buffers.
				WriteOnly = 0x01u,
				//Copy the initial buffer to merged texture
				//When the destructor of the adaptor is called, merged free-slip texture will be disintegrated and copied back to the original array of buffers.
				ReadWrite = 0x02u
			};

			//Set the memory behaviour of the free-slip texture buffer
			STPMemoryMode Mode;
			//CUDA stream for device memory allocation and copy.
			//Provide 0 to use default stream
			cudaStream_t Stream;

		};

		/**
		 * @brief STPFreeSlipLocation denotes where the free-slip data will be available.
		 * Once retrieved, the data retrieved can only be used in designated memory space
		*/
		enum class STPFreeSlipLocation : unsigned char {
			HostMemory = 0x00u,
			DeviceMemory = 0x01u
		};

	private:

		/**
		 * @brief STPHostCallbackDeleter is a deleter to return host memory to memory pool using CUDA stream callback function
		*/
		struct STPHostCallbackDeleter {
		private:

			//Stream the callback function will be enqueued
			//Host memory will be returned to
			std::optional<std::pair<cudaStream_t, STPPinnedMemoryPool*>> Data;

		public:

			STPHostCallbackDeleter() = default;

			/**
			 * @brief Init STPHostCallbackDeleter with data
			 * @param stream Stream that the free operation will be called in
			 * @param memPool The host memory pool that the memory will be returned to
			*/
			STPHostCallbackDeleter(cudaStream_t, STPPinnedMemoryPool*);

			void operator()(T*) const;

		};

		const STPFreeSlipTextureAttribute& Attr;
		const STPFreeSlipTextureData Data;

		//The free-slip texture buffer stored separately
		const STPFreeSlipTexture& Buffer;
		//The previously integrated texture location
		std::optional<STPFreeSlipLocation> Integration;
		//Pointer to merged free-slip texture in device memory.
		std::unique_ptr<T[], STPHostCallbackDeleter> HostIntegration;
		STPSmartDeviceMemory::STPStreamedDeviceMemory<T[]> DeviceIntegration;

		/**
		 * @brief Free merged buffer returned previously.
		 * If the buffer is set to be read-only, memory will be freed straight away.
		 * Otherwise the memory will be disintegrated and copied back to the original per-chunk buffer
		*/
		void destroyAllocation();

		/**
		 * @brief Calculate the number of pixel per chunk.
		 * @return The number of pixel per chunk.
		*/
		unsigned int calcChunkPixel() const;

		/**
		 * @brief Copy the buffer between free-slip buffer and each individual chunk.
		 * @tparam Pack Specifies the copy operation.
		 * True to use pack operation, for which individual chunk buffer is packed into a large free-slip buffer;
		 * False to use unpack operation, copy from the large free-slip buffer to each individual chunk buffer.
		*/
		template<bool Pack>
		void copyFreeslipBuffer();

	public:

		/**
		 * @brief Init STPFreeSlipTextureBuffer with texture buffer.
		 * All reference will be retained and caller should guarantee their life-time.
		 * @param texture An array of pointers of ranged of free-slip texture per chunk which will be used to indexed in a free-slip manner.
		 * The number of texture provided must either be 1, meaning no free-slip logic will be used on this texture,
		 * or equal to the number of total free-slip chunk count.
		 * Any other number will cause exception to be thrown.
		 * @param data The pointer to texture buffer data.
		 * @param attr Texture buffer attribute that will be used to make copy and sharing free-slip texture with other program.
		*/
		STPFreeSlipTextureBuffer(STPFreeSlipTexture&, STPFreeSlipTextureData, const STPFreeSlipTextureAttribute&);

		STPFreeSlipTextureBuffer(const STPFreeSlipTextureBuffer&) = delete;

		STPFreeSlipTextureBuffer(STPFreeSlipTextureBuffer&&) = delete;

		STPFreeSlipTextureBuffer& operator=(const STPFreeSlipTextureBuffer&) = delete;

		STPFreeSlipTextureBuffer& operator=(STPFreeSlipTextureBuffer&&) = delete;

		~STPFreeSlipTextureBuffer();

		/**
		 * @brief Get the pointer to the merged free-slip texture.
		 * No reallocation is allowed for data safety since STPFreeSlipTexture might be shared with other program, copy back to the buffer may cause race condition.
		 * Function can be called multiple times to retrieve the internal buffer
		 * @param location The memory space where the free-slip texture will be used.
		 * Only host memory will be available if memory location is host.
		 * Device memory location implies host memory, so if both memory are needed, choose device memory.
		 * @return The free-slip texture, will be available at the memory specified by location.
		 * Host memory will be available right after the function returned while device memory is stream-ordered.
		 * The underlying merged texture pointer is managed by the current texture buffer and will be freed (and optionally copied back) when the object got destroyed.
		 * If function is called repetitively, depends on the previous memory location, function will return:
		 * If the allocated location was host, host pointer is returned, the argument is ignored.
		 * If the allocated location was device, device pointer is returned if argument asks for device pointer, or host pointer otherwise.
		 * When texture is disintegrated back to the buffer, device memory will be used, such that the host memory will be read only in device mode.
		*/
		T* operator()(STPFreeSlipLocation);

		/**
		 * @brief Get the location of memory that the buffer has been allocated.
		 * If no allocation has happened, exception is thrown.
		 * @return The location of memory has been allocated
		*/
		STPFreeSlipLocation where() const;

	};

	typedef STPFreeSlipTextureBuffer<float> STPFreeSlipFloatTextureBuffer;
	typedef STPFreeSlipTextureBuffer<STPDiversity::Sample> STPFreeSlipSampleTextureBuffer;
	typedef STPFreeSlipTextureBuffer<unsigned short> STPFreeSlipRenderTextureBuffer;

}
#endif//_STP_FREESLIP_TEXTURE_BUFFER_H_