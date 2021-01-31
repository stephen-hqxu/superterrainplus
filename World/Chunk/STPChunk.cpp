#include "STPChunk.h"

using glm::vec2;
using glm::uvec2;
using glm::ivec2;
using glm::vec3;

using std::atomic_init;
using std::atomic_load_explicit;

using namespace SuperTerrainPlus;

STPChunk::STPChunk(uvec2 size, bool initialise) : PixelSize(size) {
	if (initialise) {
		const int num_pixel = size.x * size.y;
		//heightmap is RED format
		this->TerrainMaps[0] = new float[num_pixel];
		this->TerrainMaps_cache[0] = new unsigned short[num_pixel];
		//normal map is RGBA format
		this->TerrainMaps[1] = new float[num_pixel * 4];
		this->TerrainMaps_cache[1] = new unsigned short[num_pixel * 4];
	}

	atomic_init<STPChunkState>(&this->State, STPChunkState::Empty);
	atomic_init<bool>(&this->inUsed, false);
}

STPChunk::~STPChunk() {
	while (atomic_load_explicit<bool>(&this->inUsed, std::memory_order::memory_order_relaxed)) {
		//make sure the chunk is not in used, and all previous tasks are finished
	}

	if (this->TerrainMaps[0] != nullptr) {
		delete[] this->TerrainMaps[0];
	}
	if (this->TerrainMaps_cache[0] != nullptr) {
		delete[] this->TerrainMaps_cache[0];
	}

	if (this->TerrainMaps[1] != nullptr) {
		delete[] this->TerrainMaps[1];
	}
	if (this->TerrainMaps_cache[1] != nullptr) {
		delete[] this->TerrainMaps_cache[1];
	}
}

float* STPChunk::getHeightmap() {
	return this->TerrainMaps[0];
}

float* STPChunk::getNormalmap() {
	return this->TerrainMaps[1];
}

const uvec2& STPChunk::getSize() const {
	return this->PixelSize;
}

vec2 STPChunk::getChunkPosition(vec3 cameraPos, uvec2 chunkSize, float scaling) noexcept {
	const vec2 chunkSize_int = vec2(scaling * static_cast<int>(chunkSize.x), scaling * static_cast<int>(chunkSize.y));
	return vec2(glm::floor(cameraPos.x / chunkSize_int.x) * chunkSize_int.x, glm::floor(cameraPos.z / chunkSize_int.y) * chunkSize_int.y);
}

vec2 STPChunk::offsetChunk(vec2 chunkPos, uvec2 chunkSize, ivec2 offset, float scaling) noexcept {
	return chunkPos + vec2(scaling * static_cast<int>(chunkSize.x) * offset.x, scaling * static_cast<int>(chunkSize.y) * offset.y);
}

STPChunk::STPChunkPosCache STPChunk::getRegion(vec2 centerPos, uvec2 chunkSize, uvec2 regionSize, float scaling) noexcept {
	const unsigned int num_chunk = regionSize.x * regionSize.y;
	//We need to calculate the starting unit plane, i.e., the unit plane of the top-left corner of the entire rendered terrain
	//We don't need y, all the plane will be aligned at the same height, so it contains x and z position
	//Base position is negative since we want the camera locates above the center of the entire rendered plane
	const vec2 base_position = STPChunk::offsetChunk(centerPos, chunkSize, ivec2(-glm::floor(regionSize.x / 2.0f), -glm::floor(regionSize.y / 2.0f)), scaling);

	STPChunkPosCache results;
	//get the result
	for (unsigned int i = 0u; i < num_chunk; i++) {
		//Calculate the position for each chunk
		//Note that the chunk_position is not the basechunkpos, the former one is refered to the relative coordinate to the entire terrain, i.e., local chunk coordinate.
		//The latter one refers to the world coordinate

		//Basically it converts 1D chunk ID to 2D local chunk position, btw chunk position must be a positive integer
		const uvec2 local_chunk_offset = uvec2(i % regionSize.x, static_cast<unsigned int>(glm::floor<unsigned int>(i / regionSize.y)));
		//Then convert local to world coordinate
		const vec2 basePos = STPChunk::offsetChunk(base_position, chunkSize, local_chunk_offset, scaling);

		//arranged from top-left to bottom right
		results.push_back(basePos);
	}

	return results;
}