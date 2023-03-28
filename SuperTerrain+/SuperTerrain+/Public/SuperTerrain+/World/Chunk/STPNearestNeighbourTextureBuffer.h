#pragma once
#ifndef _STP_NEAREST_NEIGHBOUR_TEXTURE_BUFFER_H_
#define _STP_NEAREST_NEIGHBOUR_TEXTURE_BUFFER_H_

#include <SuperTerrain+/STPCoreDefine.h>
#include "../STPWorldMapPixelFormat.hpp"
//Neighbour Data
#include "STPNearestNeighbourInformation.hpp"
//Memory
#include "../../Utility/Memory/STPSmartDeviceMemory.h"

#include <type_traits>
#include <utility>

namespace SuperTerrainPlus {

	/**
	 * @brief STPNearestNeighbourTextureBufferMemoryMode specifies the memory operation mode for merged STPNearestNeighbourTextureBuffer.
	*/
	enum class STPNearestNeighbourTextureBufferMemoryMode : unsigned char {
		//Copy the initial texture to merged buffer.
		//When the instance is destroyed, merged neighbour buffer will be freed straight away.
		ReadOnly = 0x00u,
		//No copy from the initial texture, only allocation.
		//When the instance is destroyed, merged neighbour buffer will be disintegrated and copied back to each original chunk texture.
		WriteOnly = 0x01u,
		//Copy the initial buffer to merged texture
		//When the instance is destroyed, merged neighbour buffer will be disintegrated and copied back to each original chunk texture.
		ReadWrite = 0x02u
	};

	/**
	 * @brief STPNearestNeighbourTextureBuffer is a texture manager dedicated for nearest neighbour chunk system.
	 * It takes in a range of pointers to texture for each chunk, merge them into a large buffer aligned in row-major local-index.
	 * It stores information about each neighbour and the texture, adapts and generates texture buffer depends on usage.
	 * Note that life-time of the pointer to the merged texture buffer is controlled by the calling STPNearestNeighbourTextureBuffer.
	 * When the bounded STPNearestNeighbourTextureBuffer is destroyed, neighbour texture buffer can be optionally copied back to the original chunk.
	 * STPNearestNeighbourTextureBuffer is a light-weight class and can be created with ease.
	 * @tparam T The data type of the texture.
	 * @param MM The memory mode of the merged buffer.
	*/
	template<typename T, STPNearestNeighbourTextureBufferMemoryMode MM>
	class STP_API STPNearestNeighbourTextureBuffer {
	public:

		//The texture can only be a pointer to constant under read only mode.
		using TextureType = std::conditional_t<MM == STPNearestNeighbourTextureBufferMemoryMode::ReadOnly, std::add_const_t<T>, T>;

		//Specifies how temporary device texture buffer should be allocated and copied.
		typedef std::pair<cudaMemPool_t, cudaStream_t> STPDeviceMemoryOperator;

		/**
		 * @brief STPMemoryLocation denotes where the nearest neighbour data will be available.
		 * Once retrieved, the data retrieved can only be used in designated memory space.
		*/
		enum class STPMemoryLocation : unsigned char {
			HostMemory = 0x00u,
			DeviceMemory = 0x01u
		};

		/**
		 * @brief STPMergedBuffer is the memory returned to the application after merging nearest neighbour texture into a large buffer.
		 * This large buffer can be used like a large, row-major indexed texture.
		*/
		class STP_API STPMergedBuffer {
		public:

			//Type T but it is not constant
			using MutableT = std::remove_const_t<T>;

		private:

			const STPNearestNeighbourTextureBuffer& Main;

			//Depends on the memory location, they might be nullptr.
			STPSmartDeviceMemory::STPHost<MutableT[]> HostMem;
			STPSmartDeviceMemory::STPStreamedDevice<MutableT[]> DeviceMem;

			/**
			 * @brief Copy the texture between merged neighbour buffer and each individual chunk texture map.
			 * @tparam Pack Specifies the copy operation.
			 * True to use pack operation, for which individual chunk texture is packed into a large merged buffer;
			 * False to use unpack operation, copy from the large merged buffer to each individual chunk texture.
			*/
			template<bool Pack>
			void copyNeighbourTexture();

		public:

			//Where the merged buffer can be accessed?
			const STPMemoryLocation Location;

			/**
			 * @brief Create a merged neighbour buffer.
			 * - No more than one merged buffer should be created from the same nearest neighbour texture buffer instance,
			 *	copy between the original texture may cause race condition.
			 *	The program does not check for this, such that if user do so this is undefined behaviour.
			 * - Host memory will be available right after the construction of merged buffer, while device memory is stream-ordered.
			 *	The underlying merged texture pointer is managed and will be freed (and optionally copied back) when the object got destroyed.
			 * - Regarding memory location:
			 *	- Only host memory will be allocated if memory location is host.
			 *		The host memory will be allocated as pinned memory.
			 *	- Both device and host memory will be allocated if memory location is device.
			 *		However the host memory is only used as a cache for doing device copy internally.
			 *		When texture is copied back to the buffer, device memory will be used, such that the host memory will be read only in device mode.
			 * @param nn_texture_buffer The dependent nearest neighbour texture buffer.
			 * @param location The memory space where the merged buffer will be used.
			*/
			STPMergedBuffer(const STPNearestNeighbourTextureBuffer&, STPMemoryLocation);

			STPMergedBuffer(const STPMergedBuffer&) = delete;

			STPMergedBuffer(STPMergedBuffer&&) = delete;

			STPMergedBuffer& operator=(const STPMergedBuffer&) = delete;

			STPMergedBuffer& operator=(STPMergedBuffer&&) = delete;

			~STPMergedBuffer();

			/**
			 * @brief Get the merged neighbour buffer at host memory.
			 * If the buffer was not constructed as host memory, this memory is only intended to be used internally as copy cache.
			 * This holds the same copy as the device memory initially, if the memory mode is specified such that it can be read.
			 * Writing to the memory is allowed, but will be discarded regardless.
			 * @return A host pointer to merged buffer.
			*/
			MutableT* getHost() const noexcept;

			/**
			 * @brief Get the merged neighbour buffer at device memory.
			 * If the buffer was not constructed as device memory, return a nullptr.
			 * @return A device pointer to merged buffer.
			*/
			MutableT* getDevice() const noexcept;

		};

		const STPNearestNeighbourInformation& NeighbourInfo;
		const STPDeviceMemoryOperator DeviceMemInfo;

	private:

		//The pointer to the array of neighbour texture.
		TextureType* const* const NeighbourTexture;
		
		/**
		 * @brief Calculate the number of pixel per chunk.
		 * @return The number of pixel per chunk.
		*/
		unsigned int calcChunkPixel() const noexcept;

		/**
		 * @brief Calculate the number of pixel in total with all neighbour chunks.
		 * @return The number of pixel in the neighbour.
		*/
		unsigned int calcNeighbourPixel() const noexcept;

	public:

		/**
		 * @brief Initialise STPNearestNeighbourTextureBuffer.
		 * All reference will be retained and caller should guarantee their life-time.
		 * @param texture An array of pointers of ranged of neighbour texture per chunk.
		 * - The memory to the array is retained by the current instance and should be managed by the application,
		 *	plus the memory to each chunk texture is also required to be available.
		 *	Both memory should be valid until destruction of the current object.
		 * - The number of texture must be no less than the number of nearest neighbour chunk. Violation of this will result in UB.
		 * @param nn_info The information regarding the nearest neighbour logic. Reference is kept until destruction.
		 * @param texture_device_mem_alloc Provide information how device memory should be operated.
		 * Reference is kept until destruction.
		*/
		STPNearestNeighbourTextureBuffer(TextureType* const*, const STPNearestNeighbourInformation&, const STPDeviceMemoryOperator&);

		STPNearestNeighbourTextureBuffer(const STPNearestNeighbourTextureBuffer&) = delete;

		STPNearestNeighbourTextureBuffer(STPNearestNeighbourTextureBuffer&&) = delete;

		STPNearestNeighbourTextureBuffer& operator=(const STPNearestNeighbourTextureBuffer&) = delete;

		STPNearestNeighbourTextureBuffer& operator=(STPNearestNeighbourTextureBuffer&&) = delete;

		~STPNearestNeighbourTextureBuffer() = default;

	};

	//floating-point heightmap, write only
	typedef STPNearestNeighbourTextureBuffer<STPHeightFloat_t, STPNearestNeighbourTextureBufferMemoryMode::WriteOnly>
		STPNearestNeighbourHeightFloatWTextureBuffer;
	//floating-point heightmap, read write
	typedef STPNearestNeighbourTextureBuffer<STPHeightFloat_t, STPNearestNeighbourTextureBufferMemoryMode::ReadWrite>
		STPNearestNeighbourHeightFloatRWTextureBuffer;
	//sample type biomemap, read only
	typedef STPNearestNeighbourTextureBuffer<STPSample_t, STPNearestNeighbourTextureBufferMemoryMode::ReadOnly>
		STPNearestNeighbourSampleRTextureBuffer;
	//fixed-point heightmap, write only
	typedef STPNearestNeighbourTextureBuffer<STPHeightFixed_t, STPNearestNeighbourTextureBufferMemoryMode::WriteOnly>
		STPNearestNeighbourHeightFixedWTextureBuffer;

}
#endif//_STP_NEAREST_NEIGHBOUR_TEXTURE_BUFFER_H_