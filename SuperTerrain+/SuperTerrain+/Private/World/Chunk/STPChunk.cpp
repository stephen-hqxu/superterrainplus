#include <SuperTerrain+/World/Chunk/STPChunk.h>

#include <SuperTerrain+/Exception/STPNumericDomainError.h>

#include <glm/common.hpp>

using glm::uvec2;
using glm::ivec2;
using glm::dvec2;
using glm::dvec3;

using std::unique_ptr;
using std::make_unique;

using namespace SuperTerrainPlus;
using STPDiversity::Sample;

STPChunk::STPChunk(const uvec2 size) : MapDimension(size), Completeness(STPChunkCompleteness::Empty) {
	const unsigned int num_pixel = this->MapDimension.x * this->MapDimension.y;
	STP_ASSERTION_NUMERIC_DOMAIN(num_pixel > 0u, "The dimension of texture must not be zero");

	//allocate memory for each map
	this->Biomemap = make_unique<Sample[]>(num_pixel);
	this->Heightmap = make_unique<float[]>(num_pixel);
	this->LowHeightmap = make_unique<unsigned short[]>(num_pixel);
}

Sample* STPChunk::biomemap() noexcept {
	return this->Biomemap.get();
}

const Sample* STPChunk::biomemap() const noexcept {
	return this->Biomemap.get();
}

float* STPChunk::heightmap() noexcept {
	return this->Heightmap.get();
}

const float* STPChunk::heightmap() const noexcept {
	return this->Heightmap.get();
}

unsigned short* STPChunk::heightmapLow() noexcept {
	return this->LowHeightmap.get();
}

const unsigned short* STPChunk::heightmapLow() const noexcept {
	return this->LowHeightmap.get();
}

const ivec2* STPChunk::STPChunkNeighbourOffset::begin() const noexcept {
	return this->NeighbourOffset.get();
}

const ivec2* STPChunk::STPChunkNeighbourOffset::end() const noexcept {
	return this->NeighbourOffset.get() + this->NeighbourOffsetCount;
}

const ivec2& STPChunk::STPChunkNeighbourOffset::operator[](size_t idx) const noexcept {
	return this->NeighbourOffset[idx];
}

ivec2 STPChunk::calcWorldChunkCoordinate(const dvec3& pointPos, const uvec2& chunkSize, const dvec2& scale) noexcept {
	//scale the chunk
	const dvec2 scaled_chunk_size = static_cast<dvec2>(chunkSize) * scale;
	//determine which chunk unit the viewer is in, basically we are trying to round down to the chunk size.
	const ivec2 chunk_unit = static_cast<ivec2>(glm::floor(dvec2(pointPos.x, pointPos.z) / scaled_chunk_size));
	return chunk_unit * static_cast<ivec2>(chunkSize);
}

uvec2 STPChunk::calcLocalChunkCoordinate(const unsigned int chunkID, const uvec2& chunkRange) noexcept {
	return uvec2(chunkID % chunkRange.x, chunkID / chunkRange.x);
}

ivec2 STPChunk::calcLocalChunkOrigin(const ivec2& centreChunkCoord, const uvec2& chunkSize, const uvec2& neighbourSize) noexcept {
	//division by 2 to get the border width of the render distance, excluding the centre chunk
	//moving towards the origin, need to use negative offset
	return STPChunk::offsetChunk(centreChunkCoord, chunkSize, -static_cast<ivec2>(neighbourSize / 2u));
}

dvec2 STPChunk::calcChunkMapOffset(const ivec2& chunkCoord, const uvec2& chunkSize, const uvec2& mapSize, const dvec2& mapOffset) noexcept {
	//chunk coordinate is a multiple of chunk size
	const dvec2 chunk_unit = static_cast<dvec2>(chunkCoord) / static_cast<dvec2>(chunkSize);
	return chunk_unit * static_cast<dvec2>(mapSize) + mapOffset;
}

ivec2 STPChunk::offsetChunk(const ivec2& chunkCoord, const uvec2& chunkSize, const ivec2& offset) noexcept {
	return chunkCoord + static_cast<ivec2>(chunkSize) * offset;
}

STPChunk::STPChunkNeighbourOffset STPChunk::calcChunkNeighbourOffset(const uvec2& chunkSize, const uvec2& regionSize) {
	//prepare memory
	const unsigned int offset_count = regionSize.x * regionSize.y;
	unique_ptr<ivec2[]> offset_array = make_unique<ivec2[]>(offset_count);

	//We need to calculate the top-left corner of the entire neighbour
	//We don't need y, all the plane will be aligned at the same height, so it contains x and z position
	//We want the centre of centre be the centre of the entire neighbour region
	//The centre chunk has relative offset of zero
	const ivec2 base_position = STPChunk::calcLocalChunkOrigin(ivec2(0), chunkSize, regionSize);
	for (unsigned int i = 0u; i < offset_count; i++) {
		//Basically it converts 1D chunk ID to a 2D local chunk position
		const uvec2 local_chunk_offset = STPChunk::calcLocalChunkCoordinate(i, regionSize);
		//Multiply by the chunk size to get the chunk coordinate system
		offset_array[i] = STPChunk::offsetChunk(base_position, chunkSize, local_chunk_offset);
	}
	return STPChunkNeighbourOffset { std::move(offset_array), offset_count };
}