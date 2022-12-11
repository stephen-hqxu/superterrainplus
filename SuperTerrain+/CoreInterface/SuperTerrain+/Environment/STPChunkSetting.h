#pragma once
#ifndef _STP_CHUNK_SETTING_H_
#define _STP_CHUNK_SETTING_H_

#include <SuperTerrain+/STPCoreDefine.h>
//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPChunkSettings stores settings for each generated chunk. It will be mainly used by 2d terrain generator
	*/
	struct STP_API STPChunkSetting {
	public:

		//Determine the X*Y number of unit planes, greater chunk size gives more detailed mesh quality but requires more memory and processing power.
		//It's highly recommend to use squared chunk size.
		glm::uvec2 ChunkSize;
		//Determine the size of all generated maps, it's recommend to have the same ratio as the chunk size to avoid any possible up/down scaling on texture.
		glm::uvec2 MapSize;
		//Determine the offset of all the chunks in (x, y, z) direction.
		//This will offset the chunk loading and unloading boundary as the viewer moves.
		glm::dvec3 ChunkOffset;
		//Determine the scale of each chunk, in (x, z) direction.
		glm::dvec2 ChunkScale;
		//Specify the (x, z) offset of the terrain heightmap.
		glm::dvec2 MapOffset;

		//Specify the number of nearest neighbour chunk around the centre chunk during diversity generation, i.e. biomemap and multi-biome heightmap generation.
		//Based on the user implementation of diversity generator, this allows accessing information beyond the current working chunk.
		glm::uvec2 DiversityNearestNeighbour;
		//Specify the number of nearest neighbour chunk around the centre chunk during heightmap erosion and allows erosion happens outside the central chunk.
		//This allows erosion to work in a *free-slip* manner, allowing water droplet to travel beyond the centre chunk to avoid chunk edge artefact.
		glm::uvec2 ErosionNearestNeighbour;
		//Specify the number of nearest neighbour chunk around the centre chunk during generation of terrain texture splatmap.
		//This allows accessing pixels beyond the generating chunk on the rendering buffer.
		glm::uvec2 SplatNearestNeighbour;
		//Determine the X*Y number of chunk to be rendered surrounded by player, greater value gives further visible distance.
		//It's highly recommend to use the squared rendering distance.
		glm::uvec2 RenderDistance;

		void validate() const;
	};
}
#endif//_STP_CHUNK_SETTINGS_H_