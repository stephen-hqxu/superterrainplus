#pragma once
#ifndef _STP_CHUNK_H_
#define _STP_CHUNK_H_

//System ADT
#include <list>
//Threading
#include <atomic>
//GLM
#include "glm/gtc/matrix_transform.hpp"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	* @brief STPChunk stores chunk data (heightmap, normal map) for 2D terrain.
	* It also contains arithmetic operation for chunk position, making locating chunks in world easier.
	*/
	class STPChunk {
	private:

		//Store the 1 bit state of the current chunk
		enum class STPChunkState : unsigned char {
			//Empty chunk with no heightmap and normal map
			Empty = 0x00,
			//Chunk with heightmap generated
			Heightmap_Ready = 0x01,
			//Chunk with hydraulic eroded heightmap
			Erosion_Ready = 0x02,
			//Chunk with normal map generated and formatted, this is considered as complete state
			Complete = 0x03
		};

		//So only chunk related class can mutate the private values
		friend class STPChunkManager;
		friend class STPChunkProvider;

		float* TerrainMaps[2];
		const glm::uvec2 PixelSize;//All maps must have the same size

		//Cache that OpenGL can use to render directly, it's converted from 32 bit internal texture to 16 bit.
		//We need to keep the 32 bit copy for later chunk computations, e.g., chunk-chunk interpolation.
		//Storing them separately can avoid re-converting format everytime the chunks get updated
		unsigned short* TerrainMaps_cache[2];

		//Flags
		//Determine if there is another thread copied the current chunk for generation, meaning we can't use right now
		std::atomic<bool> inUsed;
		std::atomic<STPChunkState> State;

	public:

		/**
		 * @brief Init chunk.
		 * @param size - The size of all maps, all maps must have the same size.
		 * @param initialise - Set true to pre-allocate spaces for all map.
		*/
		STPChunk(glm::uvec2, bool = true);

		~STPChunk();

		//A chunk position cache that stores a list of chunk world position
		typedef std::list<glm::vec2> STPChunkPosCache;

		/**
		 * @brief Return the reference to the heightmap of this chunk
		 * @return The reference to the heightmap of this chunk
		*/
		float* getHeightmap();

		/**
		 * @brief Return the reference to the normalmap of this chunk
		 * @return The reference to the normalmap of this chunk
		*/
		float* getNormalmap();

		/**
		 * @brief Return the reference to the size of the pixels for all maps
		 * @return The reference to the size of the pixels for all maps
		*/
		const glm::uvec2& getSize();

		/**
		 * @brief Get the chunk position in world where the camera is located
		 * @param cameraPos The current camera opsition
		 * @param chunkSize The size of the chunk, that is the number of unit plane in (x,z) direction
		 * @param scaling The scaling applying on (x,z) direction, default is 1.0 (no scaling)
		 * @return Chunk position in world coordinate (x,z)
		*/
		static glm::vec2 getChunkPosition(glm::vec3, glm::uvec2, float = 1.0f) noexcept;

		/**
		 * @brief Move the chunk by chunk position
		 * @param chunkPos The current chunk position (x,z) in world coordinate
		 * @param chunkSize The size of the chunk, that is the number of unit plane in (x,z) direction
		 * @param offset The chunk position offset.
		 * @param scaling The scaling applying on (x,z) direction, default is 1.0 (no scaling)
		 * @return The offset chunk position in world coordinate
		*/
		static glm::vec2 offsetChunk(glm::vec2, glm::uvec2, glm::ivec2, float = 1.0f) noexcept;

		/**
		 * @brief Get an area of chunk coordinates, centered at a give chunk position
		 * @param centerPos The center chunk position in world coordinate
		 * @param chunkSize The size of the chunk, that is the number of unit plane in (x,z) direction
		 * @param regionSize The number of chunk in x and y direction in chunk coordinate
		 * @param scaling The scaling applying on (x,z) direction, default is 1.0 (no scaling)
		 * @return Chunk positions in world coordinate (x,z), aligning from top-left to bottom right
		*/
		static STPChunkPosCache getRegion(glm::vec2, glm::uvec2, glm::uvec2, float = 1.0f) noexcept;

	};
}
#endif //_STP_CHUNK_H_

