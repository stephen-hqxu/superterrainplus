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
	class STPSmartDeviceMemory final {
	private:

		STPSmartDeviceMemory() = delete;

		~STPSmartDeviceMemory() = delete;

		//Treat array as a regular type since cudaFree() treats array like normal pointer
		template<typename T>
		using NoArray = std::remove_all_extents_t<T>;

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

	public:

		//STPDeviceMemory is a normal device memory version of std::unique_ptr.
		//The deleter utilises cudaFree()
		template<typename T>
		using STPDeviceMemory =
			std::unique_ptr<
				STPSmartDeviceMemory::NoArray<T>,
				STPSmartDeviceMemory::STPDeviceMemoryDeleter<STPSmartDeviceMemory::NoArray<T>>
			>;

		//STPStreamedDeviceMemory is a stream-ordered device memory deleter.
		//The deleter utilises cudaFreeAsync()
		//However the caller should guarantee the availability of the stream when the memory is destroyed
		template<typename T>
		using STPStreamedDeviceMemory = 
			std::unique_ptr<
				STPSmartDeviceMemory::NoArray<T>,
				STPSmartDeviceMemory::STPStreamedDeviceMemoryDeleter<STPSmartDeviceMemory::NoArray<T>>
			>;

		//Some helper functions

		/**
		 * @brief Create a STPDeviceMemory which is a smart pointer to device memory with default device deleter
		 * @tparam T The type of the pointer
		 * @param size The number of element of T to be allocated (WARNING: NOT the size in byte)
		 * @return The smart pointer to the memory allocated
		*/
		template<typename T>
		static STPDeviceMemory<T> makeDevice(size_t = 1ull);

		/**
		 * @brief Create a STPStreamedDeviceMemory which is a smart pointer to device memory with stream-ordered device deleter
		 * @tparam T The type of the pointer
		 * @param size The number of element of T to be allocated (WARNING: NOT the size in byte)
		 * @param memPool The device memory pool that the memory will be allocated from
		 * @param stream The device stream the deleter will be called
		 * @return The streamed smart pointer to the memory allocated
		*/
		template<typename T>
		static STPStreamedDeviceMemory<T> makeStreamedDevice(cudaMemPool_t, cudaStream_t, size_t = 1ull);

	};

}
//Template definition should be included by user in the source code to avoid header contamination
#endif//_STP_SMART_DEVICE_MEMORY_H_