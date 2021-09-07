#pragma once
#ifndef _STP_SMART_DEVICE_MEMORY_H_
#define _STP_SMART_DEVICE_MEMORY_H_

//System
#include <memory>
#include <type_traits>

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPSmartDeviceMemoryUtility is some utilities for managed device memory
	*/
	class STPSmartDeviceMemoryUtility final {
	private:

		STPSmartDeviceMemoryUtility() = delete;

		~STPSmartDeviceMemoryUtility() = delete;

	public:

		//Treat array as a regular type since cudaFree() treats array like normal pointer
		template<typename T>
		using NoArray = std::conditional_t<std::is_array_v<T>, std::remove_pointer_t<std::decay_t<T>>, T>;

		//Delete device memory using cudaFree();
		template<typename T>
		struct STPDeviceMemoryDeleter {
		public:

			void operator()(T*) const;

		};

	};

	//STPSmartDeviceMemory is a device memory version of std::unique_ptr.
	//Currently it only supports non-stream ordered memory free
	template<typename T>
	using STPSmartDeviceMemory = 
		std::unique_ptr<
			STPSmartDeviceMemoryUtility::NoArray<T>, 
			STPSmartDeviceMemoryUtility::STPDeviceMemoryDeleter<STPSmartDeviceMemoryUtility::NoArray<T>>
		>;

}
//Template definition should be included by user in the source code to avoid header containmination
#endif//_STP_SMART_DEVICE_MEMORY_H_