#pragma once
#ifndef _STP_BIOME_DEF_H_
#define _STP_BIOME_DEF_H_

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {
	/**
	 * @brief STPBiome is a series of biome generation algorithm that allows user to define their own implementations
	*/
	namespace STPBiome {
		//Sample of the layer, it can be interpreted as biome id, or temp, or climate, or anything based on implementation
		typedef unsigned short Sample;
		//A seed is a random factor that is used to generate a random sequence, the same seed will guarantee the same generated sequence
		typedef unsigned long long Seed;
	}
}
#endif//_STP_BIOME_DEF_H_