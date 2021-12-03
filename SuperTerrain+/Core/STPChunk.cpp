#include <SuperTerrain+/World/Chunk/STPChunk.h>

#include <SuperTerrain+/Exception/STPBadNumericRange.h>

using glm::vec2;
using glm::uvec2;
using glm::ivec2;
using glm::vec3;

using std::unique_ptr;
using std::make_unique;
using std::atomic_init;
using std::atomic_load;
using std::atomic_store;
using std::ostream;
using std::istream;
using std::ios_base;

using namespace SuperTerrainPlus;

STPChunk::STPChunk(uvec2 size) : PixelSize(size), State(STPChunkState::Empty), inUsed(false) {
	const unsigned int num_pixel = size.x * size.y;
	if (num_pixel == 0) {
		throw STPException::STPBadNumericRange("The dimension of texture must not be zero");
	}

	//heightmap is R32F format
	this->Heightmap = make_unique<float[]>(num_pixel);
	//biomemap is R16UI format
	this->Biomemap = make_unique<STPDiversity::Sample[]>(num_pixel);
	//rendering buffer is R16 format heightmap
	this->HeightmapRenderingBuffer = make_unique<unsigned short[]>(num_pixel);
}

STPChunk::~STPChunk() {
	while (atomic_load<bool>(&this->inUsed)) {
		//make sure the chunk is not in used, and all previous tasks are finished
	}

	//array deleted by smart ptr
}

bool STPChunk::isOccupied() const {
	return atomic_load(&this->inUsed);
}

void STPChunk::markOccupancy(bool val) {
	atomic_store(&this->inUsed, val);
}

STPChunk::STPChunkState STPChunk::getChunkState() const {
	return atomic_load(&this->State);
}

void STPChunk::markChunkState(STPChunkState state) {
	atomic_store(&this->State, state);
}

float* STPChunk::getHeightmap() {
	return const_cast<float*>(const_cast<const STPChunk*>(this)->getHeightmap());
}

const float* STPChunk::getHeightmap() const {
	return this->Heightmap.get();
}

unsigned short* STPChunk::getRenderingBuffer() {
	return const_cast<unsigned short*>(const_cast<const STPChunk*>(this)->getRenderingBuffer());
}

const unsigned short* STPChunk::getRenderingBuffer() const {
	return this->HeightmapRenderingBuffer.get();
}

STPDiversity::Sample* STPChunk::getBiomemap() {
	return const_cast<STPDiversity::Sample*>(const_cast<const STPChunk*>(this)->getBiomemap());
}

const STPDiversity::Sample* STPChunk::getBiomemap() const {
	return this->Biomemap.get();
}

const uvec2& STPChunk::size() const {
	return this->PixelSize;
}

#define ASSERT_POSITION_SCALE(S) \
if(S <= 0.0f) { \
	throw STPException::STPBadNumericRange("Scale should be a positive floating point number."); \
}

vec2 STPChunk::getChunkPosition(vec3 cameraPos, uvec2 chunkSize, float scaling) {
	ASSERT_POSITION_SCALE(scaling);
	const vec2 scaled_chunkSize = static_cast<vec2>(chunkSize) * scaling, 
		cameraPos_2d = vec2(cameraPos.x, cameraPos.z);
	return glm::floor(cameraPos_2d / scaled_chunkSize) * scaled_chunkSize;
}

uvec2 STPChunk::getLocalChunkCoordinate(unsigned int chunkID, uvec2 chunkRange) {
	//checking for invalid chunkID is relatively expensive, so we don't...
	return uvec2(chunkID % chunkRange.x, chunkID / chunkRange.y);
}

vec2 STPChunk::calcChunkMapOffset(vec2 chunkPos, uvec2 chunkSize, uvec2 mapSize, vec2 mapOffset, float scaling) {
	ASSERT_POSITION_SCALE(scaling);
	return static_cast<vec2>(mapSize) * (chunkPos / (static_cast<vec2>(chunkSize) * scaling)) + mapOffset;
}

vec2 STPChunk::offsetChunk(vec2 chunkPos, uvec2 chunkSize, ivec2 offset, float scaling) {
	ASSERT_POSITION_SCALE(scaling);
	return chunkPos + scaling * static_cast<vec2>(chunkSize) * static_cast<vec2>(offset);
}

STPChunk::STPChunkPositionCache STPChunk::getRegion(vec2 centerPos, uvec2 chunkSize, uvec2 regionSize, float scaling) {
	ASSERT_POSITION_SCALE(scaling);
	const unsigned int num_chunk = regionSize.x * regionSize.y;
	//We need to calculate the starting unit plane, i.e., the unit plane of the top-left corner of the entire rendered terrain
	//We don't need y, all the plane will be aligned at the same height, so it contains x and z position
	//Base position is negative since we want the camera locates above the center of the entire rendered plane
	const vec2 base_position = STPChunk::offsetChunk(centerPos, chunkSize, -static_cast<ivec2>(regionSize / 2u), scaling);

	STPChunkPositionCache results;
	results.reserve(num_chunk);
	//get the result
	for (unsigned int i = 0u; i < num_chunk; i++) {
		//Calculate the position for each chunk
		//Note that the chunk_position is not the basechunkpos, the former one is refered to the relative coordinate to the entire terrain, i.e., local chunk coordinate.
		//The latter one refers to the world coordinate

		//Basically it converts 1D chunk ID to 2D local chunk position, btw chunk position must be a positive integer
		const uvec2 local_chunk_offset = STPChunk::getLocalChunkCoordinate(i, regionSize);
		//Then convert local to world coordinate
		//arranged from top-left to bottom right
		results.emplace_back(STPChunk::offsetChunk(base_position, chunkSize, local_chunk_offset, scaling));
	}

	return results;
}