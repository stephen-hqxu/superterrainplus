#include "STPChunkStorage.h"

using glm::vec2;
using glm::uvec2;

using std::shared_mutex;
using std::shared_lock;
using std::unique_lock;
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
	unique_lock<shared_mutex> construct_chunk(this->chunk_storage_lock);
	auto found = this->TerrainMap2D.find(chunkPos);
	if (found == this->TerrainMap2D.end()) {
		//not found
		STPChunk* chunk = new STPChunk(mapSize, true);
		this->TerrainMap2D.emplace(chunkPos, chunk);
		return make_pair(true, chunk);
	}
	//found
	return make_pair(false, found->second.get());
}

STPChunk* STPChunkStorage::getChunk(vec2 chunkPos) {
	shared_lock<shared_mutex> get_chunk(this->chunk_storage_lock);
	auto chunk = this->TerrainMap2D.find(chunkPos);
	//if not found, we return null
	return chunk == this->TerrainMap2D.end() ? nullptr : chunk->second.get();
}

bool STPChunkStorage::removeChunk(vec2 chunkPos) {
	unique_lock<shared_mutex> remove_chunk(this->chunk_storage_lock);
	return this->TerrainMap2D.erase(chunkPos) == 1;
}

void STPChunkStorage::clearChunk() {
	unique_lock<shared_mutex> clear_chunk(this->chunk_storage_lock);
	//chunks are managed by smart pointers so we don't need to delete them
	//clear the storage
	this->TerrainMap2D.clear();
}