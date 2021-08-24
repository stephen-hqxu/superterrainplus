#pragma once
#ifndef _STP_FREESLIP_LOCATION_HPP_
#define _STP_FREESLIP_LOCATION_HPP_

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
		 * @brief STPFreeSlipLocation denotes where tje free-slip data will be available.
		 * Once retrieved, the data retrieved can only be used in designated memory space
		*/
		enum class STPFreeSlipLocation : unsigned char {
			HostMemory = 0x00u,
			DeviceMemory = 0x01u
		};

	}
}
#endif//_STP_FREESLIP_LOCATION_HPP_