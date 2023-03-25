#pragma once
#ifndef _STP_SMART_DEVICE_MEMORY_H_
#define _STP_SMART_DEVICE_MEMORY_H_

//CUDA
#include <cuda_runtime.h>
//System
#include <memory>
#include <type_traits>

namespace SuperTerrainPlus {

	/**
	 * @brief STPSmartDeviceMemory is a collection of managed device memory
	*/
	namespace STPSmartDeviceMemory {

		/**
		 * @brief Internal implementation for template function of smart device memory.
		*/
		namespace STPImplementation {

			//The managed memory unit for different type of device memory.
			//We don't care about the array type, the deleter is the same for our application.
			template<class T, template<class> class Del>
			using STPMemoryManager = std::unique_ptr<T, Del<std::remove_extent_t<T>>>;

			//Delete pinned host memory using cudaFreeHost();
			template<typename T>
			struct STPHostDeleter {
			public:

				void operator()(T*) const;

			};

			//Delete device memory using cudaFree();
			template<typename T>
			struct STPDeviceDeleter {
			public:

				void operator()(T*) const;

			};

			//Delete device memory using cudaFreeAsync();
			template<typename T>
			struct STPStreamedDeviceDeleter {
			private:

				cudaStream_t Stream;

			public:

				STPStreamedDeviceDeleter(cudaStream_t = cudaStream_t {}) noexcept;

				void operator()(T*) const;

			};

		}

		//STPHost is a pinned host memory version of std::unique_ptr.
		//The deleter utilises cudaFreeHost()
		template<typename T>
		using STPHost = STPImplementation::STPMemoryManager<T, STPImplementation::STPHostDeleter>;

		//STPDevice is a normal device memory version of std::unique_ptr.
		//The deleter utilises cudaFree()
		template<typename T>
		using STPDevice = STPImplementation::STPMemoryManager<T, STPImplementation::STPDeviceDeleter>;

		//STPStreamedDevice is a stream-ordered device memory deleter.
		//The deleter utilises cudaFreeAsync()
		//However the caller should guarantee the availability of the stream when the memory is destroyed
		template<typename T>
		using STPStreamedDevice = STPImplementation::STPMemoryManager<T, STPImplementation::STPStreamedDeviceDeleter>;

		/**
		 * @brief STPPitchedDevice is a managed device memory with pitch.
		 * The deleter utilises cudaFree()
		*/
		template<typename T>
		struct STPPitchedDevice : public STPDevice<T> {
		public:

			size_t Pitch;

			/**
			 * @brief Create an empty pitched device memory.
			*/
			STPPitchedDevice() noexcept;

			/**
			 * @brief Create a new managed pitched device memory.
			 * @param ptr The pitched device pointer.
			 * @param pitch The pointer pitch.
			*/
			STPPitchedDevice(typename STPDevice<T>::pointer, size_t) noexcept;

			STPPitchedDevice(STPPitchedDevice&&) noexcept = default;

			STPPitchedDevice& operator=(STPPitchedDevice&&) noexcept = default;

			~STPPitchedDevice() = default;

		};

		//Some helper functions

		/**
		 * @brief Create a STPHost which is a smart pointer to page-locked memory with default pinned memory allocator.
		 * @tparam T The type of the pointer.
		 * @param size THe number of element of T to be allocated.
		 * @return The smart pointer to the memory allocated.
		*/
		template<typename T>
		STPHost<T> makeHost(size_t = 1u);

		/**
		 * @brief Create a STPDevice which is a smart pointer to device memory with default device deleter
		 * @tparam T The type of the pointer
		 * @param size The number of element of T to be allocated (WARNING: NOT the size in byte)
		 * @return The smart pointer to the memory allocated
		*/
		template<typename T>
		STPDevice<T> makeDevice(size_t = 1u);

		/**
		 * @brief Create a STPStreamedDevice which is a smart pointer to device memory with stream-ordered device deleter
		 * @tparam T The type of the pointer
		 * @param size The number of element of T to be allocated (WARNING: NOT the size in byte)
		 * @param memPool The device memory pool that the memory will be allocated from
		 * @param stream The device stream the deleter will be called
		 * @return The streamed smart pointer to the memory allocated
		*/
		template<typename T>
		STPStreamedDevice<T> makeStreamedDevice(cudaMemPool_t, cudaStream_t, size_t = 1u);

		/**
		 * @brief Create a STPPitchedDevice which is a smart pointer to pitched device memory with regular device deleter.
		 * @tparam T The type of the pointer.
		 * @param width The width of the memory, in the number of element.
		 * @param height The height of the memory, in the number of element.
		 * @return The smart pointer the pitched memory allocated.
		*/
		template<typename T>
		STPPitchedDevice<T> makePitchedDevice(size_t, size_t);

	}

}
#include "STPSmartDeviceMemory.inl"
#endif//_STP_SMART_DEVICE_MEMORY_H_