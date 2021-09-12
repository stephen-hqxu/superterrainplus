#include <SuperTerrain+/World/Chunk/STPChunkStorage.h>

using glm::vec2;
using glm::uvec2;

using std::make_pair;

using namespace SuperTerrainPlus;

void STPChunkStorage::STPHashvec2::hashSeed(size_t& seed, float value) const {
	seed ^= this->hasher(value) + 0x9e3779b9ull + (seed << 6ull) + (seed >> 2ull);
}

size_t STPChunkStorage::STPHashvec2::operator()(const vec2& position) const {
	size_t seed = 0ull;

	//hash_combine in Boost library
	this->hashSeed(seed, position.x);
	this->hashSeed(seed, position.y);

	return seed;
}

STPChunkStorage::STPChunkStorage() {

}

STPChunkStorage::STPChunkConstructed STPChunkStorage::construct(vec2 chunkPos, uvec2 mapSize) {
	auto [it, inserted] = this->TerrainMap2D.emplace(chunkPos, mapSize);
	return make_pair(inserted, &it->second);
}

STPChunk* STPChunkStorage::operator[](vec2 chunkPos) {
	auto chunk = this->TerrainMap2D.find(chunkPos);
	//if not found, we return null
	return chunk == this->TerrainMap2D.end() ? nullptr : &chunk->second;
}

size_t STPChunkStorage::size() const {
	return this->TerrainMap2D.size();
}

bool STPChunkStorage::remove(vec2 chunkPos) {
	return this->TerrainMap2D.erase(chunkPos) == 1;
}

void STPChunkStorage::clear() {
	//chunks are managed by smart pointers so we don't need to delete them
	//clear the storage
	this->TerrainMap2D.clear();
}