#pragma once
#ifndef _STP_CHUNK_H_
#define _STP_CHUNK_H_

#include <SuperTerrain+/STPCoreDefine.h>
#include "../STPWorldMapPixelFormat.hpp"

//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <memory>

namespace SuperTerrainPlus {

	/**
	* @brief STPChunk stores chunk data such as heightmap for 2D terrain.
	* It also contains arithmetic operation for chunk position, making locating chunks in world easier.
	*/
	class STP_API STPChunk {
	public:

		/**
		 * @brief STPChunkCompleteness indicates which stage of generation the current chunk has completed.
		 * The completeness status goes progressively from the `Empty` to the `Complete`.
		*/
		enum class STPChunkCompleteness : unsigned char {
			//Empty chunk with no terrain map information.
			Empty = 0x00u,
			//Chunk with biomemap generated.
			BiomemapReady = 0x01u,
			//Chunk with heightmap generated.
			HeightmapReady = 0x02u,
			//Chunk with heightmap eroded and a low pixel format heightmap generated.
			Complete = 0x03u
		};

	private:

		//The biomemap, each biome is said to be a *sample*, the underlying interpretation of pixel value is implementation defined.
		std::unique_ptr<STPSample_t[]> Biomemap;
		//The heightmap where each pixel is a normalised height value.
		std::unique_ptr<STPHeightFloat_t[]> HeightmapFloat;
		//A direct copy of the heightmap but in a cheaper pixel format for real-time streaming and rendering.
		std::unique_ptr<STPHeightFixed_t[]> HeightmapFixed;

	public:

		//The dimension of each chunk texture.
		const glm::uvec2 MapDimension;
		//Indicate which generation stage this chunk has completed.
		//This is usually set automatically by the generator, but can also be set by application to force a chunk
		//to rerun particular generation procedures.
		STPChunkCompleteness Completeness;

		/**
		 * @brief Init chunk. Pre-allocate spaces for all map
		 * @param size - The size of all maps, all maps must have the same size.
		*/
		STPChunk(glm::uvec2);

		STPChunk(const STPChunk&) = delete;

		STPChunk(STPChunk&&) noexcept = default;

		STPChunk& operator=(const STPChunk&) = delete;

		STPChunk& operator=(STPChunk&&) noexcept = default;

		~STPChunk() = default;

		/* ---------------------------- functions to retrieve individual maps --------------------------- */
		STPSample_t* biomemap() noexcept;
		const STPSample_t* biomemap() const noexcept;

		STPHeightFloat_t* heightmapFloat() noexcept;
		const STPHeightFloat_t* heightmapFloat() const noexcept;

		STPHeightFixed_t* heightmapFixed() noexcept;
		const STPHeightFixed_t* heightmapFixed() const noexcept;
		/* ------------------------------ helper functions for chunk logic ------------------------------ */

		/**
		 * @brief STPChunkNeighbourOffset is an array of relative offset of a given centre chunk.
		*/
		struct STP_API STPChunkNeighbourOffset {
		public:

			//an array of offset
			std::unique_ptr<const glm::ivec2[]> NeighbourOffset;
			//the size of the array
			size_t NeighbourOffsetCount;

			//fancy iterator functions
			const glm::ivec2* begin() const noexcept;
			const glm::ivec2* end() const noexcept;
			//fancy container operators
			const glm::ivec2& operator[](size_t) const noexcept;

		};

		/**
		 * @brief Get the chunk coordinate where a point in world space is located.
		 * The chunk coordinate is different from chunk world position where it is represented by the chunk size.
		 * @param pointPos The world position of the point.
		 * @param chunkSize The size of the chunk, that is the number of unit plane in (x, z) directions.
		 * @param scale The scaling applying on the (x, z) directions in world position.
		 * @return The chunk coordinate in the unit of chunk size in world space.
		*/
		static glm::ivec2 calcWorldChunkCoordinate(const glm::dvec3&, const glm::uvec2&, const glm::dvec2&) noexcept;

		/**
		 * @brief Convert local chunk index to local chunk coordinate
		 * @param chunkID The local chunk ID, starting from top-left corner as 0.
		 * @param chunkRange The number of chunk in x and z direction
		 * @return The local chunk coordinate, starting from top-left corner as (0,0).
		 * If chunkID is greater than (chunkRange.x * chunkRange.y - 1u), returned result is undefined.
		*/
		static glm::uvec2 calcLocalChunkCoordinate(unsigned int, const glm::uvec2&) noexcept;

		/**
		 * @brief Calculate the origin of the local chunk (i.e., the top-left corner chunk in the local neighbourhood).
		 * @param centreChunkCoord The chunk coordinate of the centre in this neighbourhood.
		 * @param chunkSize The size of the chunk.
		 * @param neighbourSize The number chunk in (x,z) direction in this neighbourhood.
		 * @return The coordinate of the origin chunk.
		*/
		static glm::ivec2 calcLocalChunkOrigin(const glm::ivec2&, const glm::uvec2&, const glm::uvec2&) noexcept;

		/**
		 * @brief Calculate the terrain map offset for a particular chunk, such that each successive map can seamlessly connect to the neighbour chunks.
		 * @param chunkCoord The chunk world coordinate.
		 * The chunk coordinate must be a multiple of chunk size.
		 * @param chunkSize The size of the chunk.
		 * @param mapSize The dimension of the terrain map in one chunk.
		 * @param mapOffset The global offset applied to the map.
		 * @return The map offset for the current chunk.
		*/
		static glm::dvec2 calcChunkMapOffset(const glm::ivec2&, const glm::uvec2&, const glm::uvec2&, const glm::dvec2&) noexcept;

		/**
		 * @brief Move the chunk by chunk unit.
		 * @param chunkCoord The current chunk coordinate (x,z) in world space.
		 * @param chunkSize The size of the chunk, that is the number of unit plane in (x,z) direction.
		 * @param offset The chunk position offset.
		 * @return The offset chunk position in world coordinate.
		*/
		static glm::ivec2 offsetChunk(const glm::ivec2&, const glm::uvec2&, const glm::ivec2&) noexcept;

		/**
		 * @brief Get an area of chunk coordinate offsets.
		 * @param chunkSize The size of the chunk, that is the number of unit plane in (x,z) direction.
		 * @param regionSize The number of chunk in x and y direction in chunk coordinate.
		 * @return Chunk coordinate offsets relative to the centre chunk in world coordinate (x,z), aligning from top-left to bottom right.
		 * The centre chunk has offset of (0, 0).
		*/
		static STPChunkNeighbourOffset calcChunkNeighbourOffset(const glm::uvec2&, const glm::uvec2&);

	};
}
#endif //_STP_CHUNK_H_