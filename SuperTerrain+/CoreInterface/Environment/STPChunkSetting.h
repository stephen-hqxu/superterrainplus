#pragma once
#ifndef _STP_CHUNK_SETTING_H_
#define _STP_CHUNK_SETTING_H_

#include <STPCoreDefine.h>
#include "STPSetting.hpp"
//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPEnvironment contains all configurations for each generators, like heightmap, normalmap, biomes, texture, etc.
	*/
	namespace STPEnvironment {

		/**
		 * @brief STPChunkSettings stores settings for each generated chunk. It will be mainly used by 2d terrain generator
		*/
		struct STP_API STPChunkSetting: public STPSetting {
		public:

			//Determine the the X*Y number of unit planes, greater chunk size will give more chunk details
			//It's highly recommend to use squared chunk size
			glm::uvec2 ChunkSize;
			//Determine the size of all generated maps, it's recommend to have the same ratio as the chunk size to avoid any possible up/down scaling on texture
			glm::uvec2 MapSize;
			//Determine the X*Y number of chunk to be renderered surrounded by player, greater value will give further visible distance.
			//It's highly recommend to use the squared rendering distance, and odd number which makes centering the chunk easier.
			glm::uvec2 RenderedChunk;
			//Determine the offset of all the chunks in (x,y,z) direction, usually there is no need to change the value other than (0,0,0)
			glm::vec3 ChunkOffset;
			//Determine the scale of the unit plane, in (x,z) direction
			float ChunkScaling;
			//Specify the (x,z) offset of the terrain heightmap, x and y specify the offset on x and y direction of the map
			glm::vec2 MapOffset;
			//Specify the number of chunk that will be used as free slip chunk and allows data access outside the central chunk
			//When both values are 1, it will effectively disable the neighbour chunk logic
			glm::uvec2 FreeSlipChunk;

			/**
			 * @brief Init STPChunksPara with defualt values
			*/
			STPChunkSetting();

			~STPChunkSetting() = default;

			bool validate() const override;
		};
	}

}
#endif//_STP_CHUNK_SETTINGS_H_