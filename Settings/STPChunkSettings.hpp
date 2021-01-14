#pragma once
#ifndef _STP_CHUNK_SETTINGS_HPP_
#define _STP_CHUNK_SETTINGS_HPP_

#include "STPSettings.hpp"
//GLM
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
using glm::uvec2;
using glm::vec3;

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPSettings contains all configurations for each generators, like heightmap, normalmap, biomes, texture, etc.
	*/
	namespace STPSettings {

		/**
		 * @brief STPChunkSettings stores settings for each generated chunk. It will be mainly used by 2d terrain generator
		*/
		struct STPChunkSettings: public STPSettings {
		public:

			//Determine the the X*Y number of unit planes, greater chunk size will give more chunk details
			//It's highly recommend to use squared chunk size
			uvec2 ChunkSize;
			//Determine the size of all generated maps, it's recommend to have the same ratio as the chunk size to avoid any possible up/down scaling on texture
			uvec2 MapSize;
			//Determine the X*Y number of chunk to be renderered surrounded by player, greater value will give further visible distance.
			//It's highly recommend to use the squared rendering distance, and odd number which makes centering the chunk easier.
			uvec2 RenderedChunk;
			//Determine the offset of all the chunks in (x,y,z) direction, usually there is no need to change the value other than (0,0,0)
			vec3 ChunkOffset;
			//Determine the scale of the unit plane, in (x,z) direction
			float ChunkScaling;
			//Specify the (x,y,z) offset of the terrain heightmap, x and z specify the offset on x and y direction of the map, y specify the height offset of the final result
			vec3 MapOffset;

			/**
			 * @brief Init STPChunksPara with defualt values
			*/
			STPChunkSettings(): STPSettings() {
				//fill with defaults
				this->ChunkSize = uvec2(0u);
				this->MapSize = uvec2(0u);
				this->RenderedChunk = uvec2(0u);
				this->ChunkOffset = vec3(0.0f);
				this->ChunkScaling = 1.0f;
				this->MapOffset = vec3(0.0f);
			}

			~STPChunkSettings() = default;

			bool validate() override {
				return ChunkScaling > 0.0f;
			}
		};
	}

}
#endif//_STP_CHUNK_SETTINGS_HPP_