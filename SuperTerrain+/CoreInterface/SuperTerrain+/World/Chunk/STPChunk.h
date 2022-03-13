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
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "../Diversity/STPBiomeDefine.h"

namespace SuperTerrainPlus {

	/**
	* @brief STPChunk stores chunk data (heightmap, normal map) for 2D terrain.
	* It also contains arithmetic operation for chunk position, making locating chunks in world easier.
	*/
	class STP_API STPChunk {
	public:

		/**
		 * @brief Store the 1 bit state of the current chunk.
		*/
		enum class STPChunkState : unsigned char {
			//Empty chunk with no heightmap and normal map
			Empty = 0x00u,
			//Chunk with biomemap generated
			BiomemapReady = 0x01u,
			//Chunk with heightmap generated
			HeightmapReady = 0x02u,
			//Chunk with normal map generated and formatted, this is considered as complete state
			Complete = 0x03u
		};

	private:

		/**
		 * @brief STPChunkUnoccupier is a smart "deleter" to mark a chunk as unoccupied automatically.
		*/
		struct STP_API STPChunkUnoccupier {
		public:

			void operator()(STPChunk*) const;

		};

		/**
		 * @brief STPMapVisitor is a smart chunk map data manager.
		 * A chunk instance can have multiple shared visitors, but only one unique visitor alive at a time.
		 * An presence of unique visitor will block access of all shared visitors.
		 * This can be helpful to allow one thread updating chunk data while ensuring no other threads will be using this chunk.
		 * Violation of the intended memory access specification will lead to undefined behaviour.
		 * @tparam Unique Specifies if this visitor should be granted unique access privilege.
		*/
		template<bool Unique>
		class STPMapVisitor {
		private:

			std::conditional_t<Unique, std::unique_ptr<STPChunk, STPChunkUnoccupier>, STPChunk*> Chunk;

			/**
			 * @brief Get the underlying pointer of a managed chunk map memory.
			 * For unique visitor, this operation always success.
			 * For shared visitor, exception will be generated if there is an alive unique visitor.
			 * @tparam T The type of the map.
			 * @param map The managed chunk map memory.
			 * @return THe pointer to the chunk map.
			*/
			template<typename T>
			auto* getMapSafe(const std::unique_ptr<T[]>&) const;

		public:

			/**
			 * @brief Initialise a chunk map visitor.
			 * For unique visitor, it is only allowed to have one alive visitor. Re-instantiation will cause exception to be generated.
			 * For shared visitor, this will always be successful.
			 * @param chunk The chunk to be visited.
			*/
			STPMapVisitor(STPChunk&) noexcept(!Unique);

			STPMapVisitor(const STPMapVisitor&) = default;

			STPMapVisitor(STPMapVisitor&&) noexcept = default;

			STPMapVisitor& operator=(const STPMapVisitor&) = default;

			STPMapVisitor& operator=(STPMapVisitor&&) noexcept = default;

			~STPMapVisitor() = default;

			//Get each type of map.

			float* heightmap();

			const float* heightmap() const;

			STPDiversity::Sample* biomemap();

			const STPDiversity::Sample* biomemap() const;

			unsigned short* heightmapBuffer();

			const unsigned short* heightmapBuffer() const;

			/**
			 * @brief Get the pointer to dependent chunk.
			 * @return The chunk pointer.
			*/
			STPChunk* operator->();

			/**
			 * @see operator->()
			*/
			const STPChunk* operator->() const;

		};

	public:

		using STPUniqueMapVisitor = STPMapVisitor<true>;
		using STPSharedMapVisitor = const STPMapVisitor<false>;

	private:

		std::unique_ptr<float[]> Heightmap;
		std::unique_ptr<STPDiversity::Sample[]> Biomemap;
		//Cache that OpenGL can use to render directly, it's converted from 32 bit internal texture to 16 bit.
		//We need to keep the 32 bit copy for later chunk computations, e.g., chunk-chunk interpolation.
		//Storing them separately can avoid re-converting format every time the chunks get updated
		//Rendering buffer contain R channel only for heightmap
		std::unique_ptr<unsigned short[]> HeightmapRenderingBuffer;

		//Flags
		//Determine if there is another thread copied the current chunk for generation, meaning we can't use right now
		std::atomic<bool> Occupied;
		std::atomic<STPChunkState> State;

		/**
		 * @brief Atomically change the use status of this chunk
		 * @param val The new occupancy status of this chunk
		*/
		void markOccupancy(bool);

	public:

		//The dimension of each map for this chunk.
		const glm::uvec2 PixelSize;

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
		 * @param cameraPos The current camera position
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
		 * @brief Calculate the terrain map offset for a particular chunk, such that each successive map can seamlessly connect to the neighbour chunks.
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
		 * @brief Get an area of chunk coordinates, centred at a give chunk position
		 * @param centerPos The centre chunk position in world coordinate
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
		bool occupied() const;

		/**
		 * @brief Atomically determine the current state of the chunk
		 * @return The state code of the chunk
		*/
		STPChunkState chunkState() const;

		/**
		 * @brief Atomically change the chunk state.
		 * @param state The new state of this chunk.
		*/
		void markChunkState(STPChunkState);

	};
}
#endif //_STP_CHUNK_H_

