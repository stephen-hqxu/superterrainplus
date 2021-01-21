#include "STPChunkStorage.h"

using glm::vec2;

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

bool STPChunkStorage::addChunk(vec2 chunkPos, STPChunk* chunk) {
	return this->TerrainMap2D.emplace(chunkPos, std::unique_ptr<STPChunk>(chunk)).second;
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