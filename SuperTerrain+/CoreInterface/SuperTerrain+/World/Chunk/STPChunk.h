#pragma once
#ifndef _STP_CHUNK_H_
#define _STP_CHUNK_H_

#include <SuperTerrain+/STPCoreDefine.h>
//System
#include <vector>
#include <iostream>
#include <memory>
//Threading
#include <atomic>
//GLM
#include "glm/gtc/matrix_transform.hpp"

#include "../Diversity/STPBiomeDefine.h"

namespace SuperTerrainPlus {

	/**
	* @brief STPChunk stores chunk data (heightmap, normal map) for 2D terrain.
	* It also contains arithmetic operation for chunk position, making locating chunks in world easier.
	*/
	class STP_API STPChunk {
	public:

		//Store the 1 bit state of the current chunk
		enum class STPChunkState : unsigned char {
			//Empty chunk with no heightmap and normal map
			Empty = 0x00u,
			//Chunk with biomemap generated
			Biomemap_Ready = 0x01u,
			//Chunk with heightmap generated
			Heightmap_Ready = 0x02u,
			//Chunk with hydraulic eroded heightmap
			Erosion_Ready = 0x03u,
			//Chunk with normal map generated and formatted, this is considered as complete state
			Complete = 0x04u
		};

	private:

		//The last 8 bytes of the text "SuperTerrain+STPChunk" in SHA-256
		constexpr static unsigned long long IDENTIFIER = 0x72539f230fdbf627ull;
		//Serialisation version number, 1 byte for major version, 1 byte for minor version
		//current v1.1
		constexpr static unsigned short SERIAL_VERSION = 0x0101u;

		const glm::uvec2 PixelSize;//All maps must have the same size

		std::unique_ptr<float[]> Heightmap;
		std::unique_ptr<STPDiversity::Sample[]> Biomemap;
		//Cache that OpenGL can use to render directly, it's converted from 32 bit internal texture to 16 bit.
		//We need to keep the 32 bit copy for later chunk computations, e.g., chunk-chunk interpolation.
		//Storing them separately can avoid re-converting format everytime the chunks get updated
		//Rendering buffer contain R channel only for heightmap
		std::unique_ptr<unsigned short[]> HeightmapRenderingBuffer;

		//Flags
		//Determine if there is another thread copied the current chunk for generation, meaning we can't use right now
		std::atomic<bool> inUsed;
		std::atomic<STPChunkState> State;

	public:

		/**
		 * @brief Init chunk. Pre-allocate spaces for all map
		 * @param size - The size of all maps, all maps must have the same size.
		*/
		STPChunk(glm::uvec2);

		STPChunk(const STPChunk&) = delete;

		STPChunk(STPChunk&&) noexcept = default;

		STPChunk& operator=(const STPChunk&) = delete;

		STPChunk& operator=(STPChunk&&) noexcept = default;

		~STPChunk();

		//A chunk position cache that stores a list of chunk world position
		typedef std::vector<glm::vec2> STPChunkPositionCache;

		/**
		 * @brief Get the chunk position in world where the camera is located
		 * @param cameraPos The current camera opsition
		 * @param chunkSize The size of the chunk, that is the number of unit plane in (x,z) direction
		 * @param scaling The scaling applying on (x,z) direction, default is 1.0 (no scaling)
		 * @return Chunk position in world coordinate (x,z)
		*/
		static glm::vec2 getChunkPosition(glm::vec3, glm::uvec2, float = 1.0f);

		/**
		 * @brief Convert local chunk index to local chunk coordinate
		 * @param chunkID The local chunk ID, starting from top-left corner as 0.
		 * @param chunkRange The number of chunk in x and z direction
		 * @return The local chunk coordinate, starting from top-left corner as (0,0).
		 * If chunkID is greater than (chunkRange.x * chunkRange.y - 1u), returned result is undefined.
		*/
		static glm::uvec2 getLocalChunkCoordinate(unsigned int, glm::uvec2);

		/**
		 * @brief Calculate the terrain map offset for a particular chunk, such that each sucessive map can seamlessly connect to the neighbour chunks.
		 * @param chunkPos The current chunk position (x,z) in world coordinate.
		 * @param chunkSize The size of the chunk, that is the number of unit plane in (x,z) direction
		 * @param mapSize The dimension of terrain map in one chunk.
		 * @param mapOffset The global offset of the terrain map.
		 * @param scaling The scaling applying on (x,z) direction, default is 1.0 (no scaling)
		 * @return The terrain map offset of a particular chunk.
		*/
		static glm::vec2 calcChunkMapOffset(glm::vec2, glm::uvec2, glm::uvec2, glm::vec2, float = 1.0f);

		/**
		 * @brief Move the chunk by chunk position
		 * @param chunkPos The current chunk position (x,z) in world coordinate
		 * @param chunkSize The size of the chunk, that is the number of unit plane in (x,z) direction
		 * @param offset The chunk position offset.
		 * @param scaling The scaling applying on (x,z) direction, default is 1.0 (no scaling)
		 * @return The offset chunk position in world coordinate
		*/
		static glm::vec2 offsetChunk(glm::vec2, glm::uvec2, glm::ivec2, float = 1.0f);

		/**
		 * @brief Get an area of chunk coordinates, centered at a give chunk position
		 * @param centerPos The center chunk position in world coordinate
		 * @param chunkSize The size of the chunk, that is the number of unit plane in (x,z) direction
		 * @param regionSize The number of chunk in x and y direction in chunk coordinate
		 * @param scaling The scaling applying on (x,z) direction, default is 1.0 (no scaling)
		 * @return Chunk positions in world coordinate (x,z), aligning from top-left to bottom right
		*/
		static STPChunkPositionCache getRegion(glm::vec2, glm::uvec2, glm::uvec2, float = 1.0f);

		/**
		 * @brief Atomically determine if current chunk is used by other threads
		 * @return True if there are already thread grabbing this chunk
		*/
		bool isOccupied() const;

		/**
		 * @brief Atomically change the use status of this chunk
		 * @param val The new occupancy status of this chunk
		*/
		void markOccupancy(bool);

		/**
		 * @brief Atomically determine the current state of the chunk
		 * @return The state code of the chunk
		*/
		STPChunkState getChunkState() const;

		/**
		 * @brief Atomically change the chunk state
		 * @param state The new state of this chunk
		*/
		void markChunkState(STPChunkState);

		/**
		 * @brief Return the reference to the 32bit heightmap of this chunk
		 * @return The reference to the heightmap
		*/
		float* getHeightmap();

		/**
		 * @brief Return the reference to the 32bit heightmap of this chunk
		 * @return The reference to the heightmap
		*/
		const float* getHeightmap() const;

		/**
		 * @brief Return the reference to the 16bit integer map for rendering of this chunk
		 * @return The reference to the rendering map
		*/
		unsigned short* getRenderingBuffer();

		/**
		 * @brief Return the reference to the 16bit integer map for rendering of this chunk
		 * @return The reference to the rendering map
		*/
		const unsigned short* getRenderingBuffer() const;

		/**
		 * @brief Return the reference to the biomemap of this chunk
		 * @return The reference to the biomemap
		*/
		STPDiversity::Sample* getBiomemap();

		/**
		 * @brief Return the reference to the biomemap of this chunk
		 * @return The reference to the biomemap
		*/
		const STPDiversity::Sample* getBiomemap() const;

		/**
		 * @brief Return the reference to the size of the pixels for all maps
		 * @return The reference to the size of the pixels for all maps
		*/
		const glm::uvec2& size() const;

	};
}
#endif //_STP_CHUNK_H_

