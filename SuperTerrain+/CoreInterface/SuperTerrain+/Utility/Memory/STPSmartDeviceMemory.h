#pragma once
#ifndef _STP_SMART_DEVICE_MEMORY_H_
#define _STP_SMART_DEVICE_MEMORY_H_

//CUDA
#include <cuda_runtime.h>
//System
#include <memory>
#include <type_traits>
#include <optional>

namespace SuperTerrainPlus {

	/**
	 * @brief STPSmartDeviceMemory is a collection of managed device memory
	*/
	namespace STPSmartDeviceMemory {

		/**
		 * @brief Inline implementation for template function of smart device memory.
		*/
		namespace STPImplementation {

			//Treat array as a regular type since cudaFree() treats array like normal pointer
			template<typename T>
			using NoArray = std::remove_all_extents_t<T>;

			//The managed memory unit for different type of device memory.
			template<class T, template<class> class Del>
			using STPMemoryManager = std::unique_ptr<STPImplementation::NoArray<T>, Del<STPImplementation::NoArray<T>>>;

			//Delete pinned host memory using cudaFreeHost();
			template<typename T>
			struct STPPinnedMemoryDeleter {
			public:

				void operator()(T*) const;

			};

			//Delete device memory using cudaFree();
			template<typename T>
			struct STPDeviceMemoryDeleter {
			public:

				void operator()(T*) const;

			};

			//Delete device memory using cudaFreeAsync();
			template<typename T>
			struct STPStreamedDeviceMemoryDeleter {
			private:

				std::optional<cudaStream_t> Stream;

			public:

				STPStreamedDeviceMemoryDeleter() = default;

				STPStreamedDeviceMemoryDeleter(cudaStream_t);

				void operator()(T*) const;

			};

		}

		//STPPinnedMemory is a pinned host memory version of std::unique_ptr.
		//The deleter utilises cudaFreeHost()
		template<typename T>
		using STPPinnedMemory = STPImplementation::STPMemoryManager<T, STPImplementation::STPPinnedMemoryDeleter>;

		//STPDeviceMemory is a normal device memory version of std::unique_ptr.
		//The deleter utilises cudaFree()
		template<typename T>
		using STPDeviceMemory = STPImplementation::STPMemoryManager<T, STPImplementation::STPDeviceMemoryDeleter>;

		//STPStreamedDeviceMemory is a stream-ordered device memory deleter.
		//The deleter utilises cudaFreeAsync()
		//However the caller should guarantee the availability of the stream when the memory is destroyed
		template<typename T>
		using STPStreamedDeviceMemory = STPImplementation::STPMemoryManager<T, STPImplementation::STPStreamedDeviceMemoryDeleter>;

		/**
		 * @brief STPPitchedDeviceMemory is a managed device memory with pitch.
		 * The deleter utilises cudaFree()
		*/
		template<typename T>
		struct STPPitchedDeviceMemory : public STPDeviceMemory<T> {
		public:

			size_t Pitch;

			/**
			 * @brief Create an empty pitched device memory.
			*/
			STPPitchedDeviceMemory();

			/**
			 * @brief Create a new managed pitched device memory.
			 * @param ptr The pitched device pointer.
			 * @param pitch The pointer pitch.
			*/
			STPPitchedDeviceMemory(STPImplementation::NoArray<T>*, size_t);

			STPPitchedDeviceMemory(STPPitchedDeviceMemory&&) noexcept = default;

			STPPitchedDeviceMemory& operator=(STPPitchedDeviceMemory&&) noexcept = default;

			~STPPitchedDeviceMemory() = default;

		};

		//Some helper functions

		/**
		 * @brief Create a STPPinnedMemory which is a smart pointer to page-locked memory with default pinned memory allocator.
		 * @tparam T The type of the pointer.
		 * @param size THe number of element of T to be allocated.
		 * @return The smart pointer to the memory allocated.
		*/
		template<typename T>
		STPPinnedMemory<T> makePinned(size_t = 1u);

		/**
		 * @brief Create a STPDeviceMemory which is a smart pointer to device memory with default device deleter
		 * @tparam T The type of the pointer
		 * @param size The number of element of T to be allocated (WARNING: NOT the size in byte)
		 * @return The smart pointer to the memory allocated
		*/
		template<typename T>
		STPDeviceMemory<T> makeDevice(size_t = 1u);

		/**
		 * @brief Create a STPStreamedDeviceMemory which is a smart pointer to device memory with stream-ordered device deleter
		 * @tparam T The type of the pointer
		 * @param size The number of element of T to be allocated (WARNING: NOT the size in byte)
		 * @param memPool The device memory pool that the memory will be allocated from
		 * @param stream The device stream the deleter will be called
		 * @return The streamed smart pointer to the memory allocated
		*/
		template<typename T>
		STPStreamedDeviceMemory<T> makeStreamedDevice(cudaMemPool_t, cudaStream_t, size_t = 1u);

		/**
		 * @brief Create a STPPitchedDeviceMemory which is a smart pointer to pitched device memory with regular device deleter.
		 * @tparam T The type of the pointer.
		 * @param width The width of the memory, in the number of element.
		 * @param height The height of the memory, in the number of element.
		 * @return The smart pointer the pitched memory allocated.
		*/
		template<typename T>
		STPPitchedDeviceMemory<T> makePitchedDevice(size_t, size_t);

	}

}
#include "STPSmartDeviceMemory.inl"
#endif//_STP_SMART_DEVICE_MEMORY_H_