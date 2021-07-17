#include "STPChunkProvider.h"

using glm::vec2;
using glm::uvec2;

using std::list;
using std::make_unique;
using std::make_pair;

using namespace SuperTerrainPlus;

STPChunkProvider::STPChunkProvider(const STPSettings::STPChunkSettings& chunk_settings, STPChunkStorage& storage, STPCompute::STPHeightfieldGenerator& heightfield_generator)
	: ChunkSettings(chunk_settings), ChunkStorage(storage), generateHeightfield(heightfield_generator) {
	this->kernel_launch_pool = make_unique<STPThreadPool>(5u);
}

unsigned int STPChunkProvider::calculateMaxConcurrency(uvec2 rendered_range, uvec2 freeslip_range) {
	using glm::floor;
	const vec2 rr = vec2(rendered_range),
		fr = vec2(freeslip_range);
	const uvec2 max_used = uvec2(floor((rr + floor(fr / 2.0f) * 2.0f) / fr));
	return max_used.x * max_used.y;
}

float2 STPChunkProvider::calcChunkOffset(vec2 chunkPos) const {
	//first convert chunk world position to relative chunk position, then multiply by the map size, such that the generated map will be seamless
	return make_float2(
		static_cast<float>(this->ChunkSettings.MapSize.x) * chunkPos.x / (static_cast<float>(this->ChunkSettings.ChunkSize.x) * this->ChunkSettings.ChunkScaling) + this->ChunkSettings.MapOffset.x,
		static_cast<float>(this->ChunkSettings.MapSize.y) * chunkPos.y / (static_cast<float>(this->ChunkSettings.ChunkSize.y) * this->ChunkSettings.ChunkScaling) + this->ChunkSettings.MapOffset.y
	);
}

void STPChunkProvider::computeHeightmap(STPChunk* current_chunk, vec2 chunkPos) {
	using namespace STPCompute;

	STPHeightfieldGenerator::STPMapStorage maps;
	maps.Heightmap32F.reserve(1ull);
	maps.Heightmap32F.push_back(current_chunk->getHeightmap());
	maps.HeightmapOffset = this->calcChunkOffset(chunkPos);
	const STPHeightfieldGenerator::STPGeneratorOperation op = STPHeightfieldGenerator::HeightmapGeneration;

	//computing, check success state
	try {
		this->generateHeightfield(maps, op);
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		exit(-1);
	}
}

void STPChunkProvider::computeErosion(STPChunk* current_chunk, list<STPChunk*>& neighbour_chunks) {
	using namespace STPCompute;

	STPHeightfieldGenerator::STPMapStorage maps;
	maps.Heightmap32F.reserve(neighbour_chunks.size());
	maps.Heightfield16UI.reserve(neighbour_chunks.size());
	for (STPChunk* chk : neighbour_chunks) {
		maps.Heightmap32F.push_back(chk->getHeightmap());
		maps.Heightfield16UI.push_back(chk->getRenderingBuffer());
	}
	const STPHeightfieldGenerator::STPGeneratorOperation op =
		STPHeightfieldGenerator::Erosion |
		STPHeightfieldGenerator::RenderingBufferGeneration;

	//computing and return success state
	try {
		this->generateHeightfield(maps, op);
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		exit(-1);
	}
}

bool STPChunkProvider::checkChunk(vec2 chunkPos, std::function<bool(glm::vec2)> reload_callback) {
	auto heightmap_computer = [this](STPChunk* chunk, vec2 position) -> void {
		this->computeHeightmap(chunk, position);
		//computation was successful
		chunk->markChunkState(STPChunk::STPChunkState::Heightmap_Ready);
		chunk->markOccupancy(false);
	};
	auto erosion_computer = [this](STPChunk* centre, list<STPChunk*> neighbours) -> void {
		this->computeErosion(centre, neighbours);
		//erosion was successful
		//mark center chunk complete
		centre->markChunkState(STPChunk::STPChunkState::Complete);
		//unlock all neighbours
		for (STPChunk* chk : neighbours) {
			chk->markOccupancy(false);
		}
	};

	STPChunk* center = this->ChunkStorage.getChunk(chunkPos);
	if (center != nullptr && center->getChunkState() == STPChunk::STPChunkState::Complete) {
		//no need to continue if center chunk is available
		//since the center chunk might be used as a neighbour chunk later, we only return bool instead of a pointer
		//after checkChunk() is performed for every chunks, we can grab all pointers and check for occupancy in other functions.
		return true;
	}
	//reminder: central chunk is included in neighbours
	const STPSettings::STPChunkSettings& chk_config = this->getChunkSettings();
	const STPChunk::STPChunkPositionCache neighbour_position = STPChunk::getRegion(chunkPos, chk_config.ChunkSize, chk_config.FreeSlipChunk, chk_config.ChunkScaling);
	
	bool canContinue = true;
	//The first pass: check if all neighbours are heightmap-complete
	list<STPChunk*> neighbour;
	for (vec2 neighbourPos : neighbour_position) {
		//get current neighbour chunk
		STPChunkStorage::STPChunkConstructed res = this->ChunkStorage.constructChunk(neighbourPos, chk_config.MapSize);
		STPChunk* curr_neighbour = res.second;
		if (res.first) {
			//neighbour doesn't exist and has been added
			curr_neighbour->markOccupancy(true);
			this->kernel_launch_pool->enqueue_void(heightmap_computer, curr_neighbour, neighbourPos);
			//try to compute all heightmap, and when heightmap is computing, we don't need to wait
			canContinue = false;
		}
		neighbour.push_back(curr_neighbour);
		//if chunk is found, we can guarantee it's in-used empty or at least heightmap complete
	}
	if (!canContinue) {
		//if heightmap is computing, we don't need to check for erosion because some chunks are in use
		return false;
	}

	//The second pass: launch full compute
	for (STPChunk* chk : neighbour) {
		//if any of the chunk is occupied, we cannot continue
		if (chk->isOccupied()) {
			return false;
		}
	}
	//all chunks are available
	for (STPChunk* chk : neighbour) {
		chk->markOccupancy(true);
	}
	//send the list of neighbour chunks to GPU to perform free-slip hydraulic erosion
	this->kernel_launch_pool->enqueue_void(erosion_computer, this->ChunkStorage.getChunk(chunkPos), neighbour);
	//trigger a chunk reload as some chunks have been added to render buffer already after neighbours are updated
	for (vec2 position : neighbour_position) {
		reload_callback(position);
	}

	return true;
}

STPChunk* STPChunkProvider::requestChunk(vec2 chunkPos) {
	//after calling checkChunk(), we can guarantee it's not null
	STPChunk* chunk = this->ChunkStorage.getChunk(chunkPos);
	if (chunk != nullptr) {
		if (!chunk->isOccupied() && chunk->getChunkState() == STPChunk::STPChunkState::Complete) {
			//since we wait for all threads to finish checkChunk(), such that occupancy status will not be changed here
			return chunk;
		}
		return nullptr;
	}
	throw std::runtime_error("Chunk chunk should have been computed but not found");
}

const STPSettings::STPChunkSettings& STPChunkProvider::getChunkSettings() const {
	return this->ChunkSettings;
}
