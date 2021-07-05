#include "STPChunkStorage.h"

using glm::vec2;
using glm::uvec2;

using std::unique_ptr;
using std::make_unique;
using std::make_pair;

using namespace SuperTerrainPlus;

size_t STPChunkStorage::STPHashvec2::operator()(const vec2& position) const {
	return std::hash<float>()(position.x) ^ std::hash<float>()(position.y);
}

STPChunkStorage::STPChunkStorage() {

}

STPChunkStorage::~STPChunkStorage() {
	//free memory
	this->clearChunk();
}

STPChunkStorage::STPChunkConstructed STPChunkStorage::constructChunk(vec2 chunkPos, uvec2 mapSize) {
	auto found = this->TerrainMap2D.find(chunkPos);
	if (found == this->TerrainMap2D.end()) {
		//not found
		unique_ptr<STPChunk> chunk = make_unique<STPChunk>(mapSize, true);
		STPChunk* new_chunk = chunk.get();
		this->TerrainMap2D.emplace(chunkPos, std::move(chunk));
		return make_pair(true, new_chunk);
	}
	//found
	return make_pair(false, found->second.get());
}

STPChunk* STPChunkStorage::getChunk(vec2 chunkPos) {
	auto chunk = this->TerrainMap2D.find(chunkPos);
	//if not found, we return null
	return chunk == this->TerrainMap2D.end() ? nullptr : chunk->second.get();
}

bool STPChunkStorage::removeChunk(vec2 chunkPos) {
	return this->TerrainMap2D.erase(chunkPos) == 1;
}

void STPChunkStorage::clearChunk() {
	//chunks are managed by smart pointers so we don't need to delete them
	//clear the storage
	this->TerrainMap2D.clear();
}