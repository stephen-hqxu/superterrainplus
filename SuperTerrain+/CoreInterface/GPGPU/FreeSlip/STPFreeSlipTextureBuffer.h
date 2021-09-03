#pragma once
#ifndef _STP_FREESLIP_TEXTURE_BUFFER_H_
#define _STP_FREESLIP_TEXTURE_BUFFER_H_

#include <STPCoreDefine.h>
//System
#include <optional>
//Data Structure
#include <vector>
#include <tuple>
#include "../../Utility/STPMemoryPool.h"
//CUDA
#include <cuda_runtime.h>
//Free-Slip Data
#include "STPFreeSlipLocation.hpp"
#include "../../World/Diversity/STPBiomeDefine.h"

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
		 * @brief STPFreeSlipTextureAttribute stores invariants for free-slip texture buffer management.
		 * It should be preserved by the caller so creation of STPFreeSlipTextureBuffer is light-weight.
		*/
		struct STPFreeSlipTextureAttribute {
		public:

			//Denotes the number of pixel in one texture of a chunk
			size_t TexturePixel;

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
			 * @brief STPFreeSlipTextureData contanins essential data for texture buffer
			*/
			struct STPFreeSlipTextureData {
			public:

				/**
				 * @brief STPMemoryMode contains set of rules for how the free-slip memory manager to react
				*/
				enum class STPMemoryMode : unsigned char {
					//Copy the initial buffer to merged texture
					//When the destructor of the adaptor is called, merged free-slip texture will be freed straighet away
					ReadOnly = 0x00u,
					//No copy from the initial buffer, only allocation
					//When the destructor of the adaptor is called, merged free-slip texture will be disintegrated and copied back to the original array of buffers.
					WriteOnly = 0x01u,
					//Copy the initial buffer to merged texture
					//When the destructor of the adaptor is called, merged free-slip texture will be disintegrated and copied back to the original array of buffers.
					ReadWrite = 0x02u
				};

				//The number of channel per pixel
				unsigned char Channel;
				//Set the memory behaviour of the free-slip texture buffer
				STPMemoryMode Mode;

				//CUDA stream for device memory allocation and copy.
				//Provide 0 to use default stream
				cudaStream_t Stream;

			};

		private:

			const STPFreeSlipTextureAttribute& Attr;
			const STPFreeSlipTextureData Data;

			//The free-slip texture buffer stored separately
			STPFreeSlipTexture& Buffer;
			//The previously integrated texture and its location
			//0: Pointer to merged free-slip texture in host pinned memory.
			//1: Pointer to merged free-slip texture in device memory
			//2: The location of the pointer
			std::optional<std::tuple<T*, T*, STPFreeSlipLocation>> Integration;

			/**
			 * @brief Free merged buffer returned previously.
			 * If the buffer is set to be read-only, memory will be freed straight away.
			 * Otherwise the memory will be disintegrated and copied back to the original per-chunk buffer
			*/
			void destroyAllocation();

		public:

			/**
			 * @brief Init STPFreeSlipTextureBuffer with texture buffer.
			 * All reference will be retained and caller should guarantee their life-time.
			 * @param texture The pointer to the pointers of ranged of free-slip texture per chunk which will be used to indexed in a free-slip manner.
			 * @param data The pointer to texture buffer data, it will be copied to the current object
			 * @param attr Texture buffer attribute that will be used to make copy and sharing free-slip texture with other program
			*/
			STPFreeSlipTextureBuffer(typename STPFreeSlipTexture&, const STPFreeSlipTextureData&, const STPFreeSlipTextureAttribute&);

			STPFreeSlipTextureBuffer(const STPFreeSlipTextureBuffer&) = delete;

			STPFreeSlipTextureBuffer(STPFreeSlipTextureBuffer&&) = delete;

			STPFreeSlipTextureBuffer& operator=(const STPFreeSlipTextureBuffer&) = delete;

			STPFreeSlipTextureBuffer& operator=(STPFreeSlipTextureBuffer&&) = delete;

			~STPFreeSlipTextureBuffer() noexcept(false);
			
			/**
			 * @brief Get the pointer to the merged free-slip texture.
			 * No reallocation is allowed for data safety since STPFreeSlipTexture might be shared with other program, copy back to the buffer may cause race condition.
			 * Function can be called multiple times to retrieve the internal buffer
			 * @param location The memory space where the free-slip texture will be used.
			 * Only host memory will be available if memory location is host.
			 * Device memory location impiles host memory, so if both memory are needed, choose device memory.
			 * @return The free-slip texture, will be available at the memory specified by location.
			 * Host memory will be available right after the function returned while device memory is stream-ordered.
			 * The underlying merged texture pointer is managed by the current texture buffer and will be freed (and optionally copied back) when the object got destroyed.
			 * If function is called repetitively, depends on the previous memory location, function will return:
			 * If the allocated location was host, host pointer is returned, the argument is ignored.
			 * If the allocated location was device, device pointer is returned if argument asks for device pointer, or host pointer otherwise.
			*/
			T* operator()(STPFreeSlipLocation);

			/**
			 * @brief Get the location of memory that the buffer has been allocated.
			 * If no allocation has happned, exception is thrown
			*/
			operator STPFreeSlipLocation() const;

		};

		typedef STPFreeSlipTextureBuffer<float> STPFreeSlipFloatTextureBuffer;
		typedef STPFreeSlipTextureBuffer<STPDiversity::Sample> STPFreeSlipSampleTextureBuffer;
		typedef STPFreeSlipTextureBuffer<unsigned short> STPFreeSlipRenderTextureBuffer;

	}
}
#endif//_STP_FREESLIP_TEXTURE_BUFFER_H_