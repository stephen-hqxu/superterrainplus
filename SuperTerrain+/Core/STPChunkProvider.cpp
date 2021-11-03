#include <SuperTerrain+/World/Chunk/STPChunkProvider.h>

#include <SuperTerrain+/Utility/Exception/STPMemoryError.h>
#include <SuperTerrain+/Utility/Exception/STPInvalidEnvironment.h>
#include <SuperTerrain+/Utility/Exception/STPAsyncGenerationError.h>

#include <algorithm>
#include <sstream>

using glm::ivec2;
using glm::vec2;
using glm::uvec2;
using glm::ivec3;
using glm::round;

using std::list;
using std::make_unique;
using std::make_pair;
using std::for_each;

using std::current_exception;
using std::unique_lock;
using std::shared_lock;
using std::shared_mutex;

using namespace SuperTerrainPlus;

STPChunkProvider::STPChunkProvider(const STPEnvironment::STPChunkSetting& chunk_settings, STPChunkStorage& storage,
	STPDiversity::STPBiomeFactory& biome_factory, STPCompute::STPHeightfieldGenerator& heightfield_generator)
	: ChunkSetting(chunk_settings), ChunkStorage(storage), generateBiome(biome_factory), generateHeightfield(heightfield_generator), kernel_launch_pool(5u) {
	if (!chunk_settings.validate()) {
		throw STPException::STPInvalidEnvironment("Values from STPChunkSetting are not validated");
	}
}

STPChunkProvider::~STPChunkProvider() {
	while (this->kernel_launch_pool.size() > 0ull) {
		//sync and wait for the thread pool to finish all compute threads
	}
}

unsigned int STPChunkProvider::calculateMaxConcurrency(uvec2 rendered_range, uvec2 freeslip_range) {
	//multiply by 4: free slip calculation comes with two stages: biomemap and heightmap
	//generate a heightmap require free-slip biomemap, and eroding heightmap require free-slip heightmap, so we come with "double banding"
	const uvec2 max_used = uvec2((rendered_range + (freeslip_range / 2u) * 4u) / freeslip_range);
	return max_used.x * max_used.y;
}

inline vec2 STPChunkProvider::calcChunkOffset(vec2 chunkPos) const {
	const STPEnvironment::STPChunkSetting& chk_config = this->ChunkSetting;
	return STPChunk::calcChunkMapOffset(chunkPos, chk_config.ChunkSize, chk_config.MapSize, chk_config.MapOffset, chk_config.ChunkScaling);
}

STPChunk::STPChunkPositionCache STPChunkProvider::getNeighbour(vec2 chunkPos) const {
	const STPEnvironment::STPChunkSetting& chk_config = this->getChunkSetting();
	return STPChunk::getRegion(chunkPos, chk_config.ChunkSize, chk_config.FreeSlipChunk, chk_config.ChunkScaling);
}

void STPChunkProvider::computeHeightmap(STPChunk* current_chunk, STPChunkProvider::STPChunkNeighbour& neighbour_chunks, vec2 chunkPos) {
	//generate heightmap
	using namespace STPCompute;
	STPHeightfieldGenerator::STPMapStorage maps;
	maps.Biomemap.reserve(neighbour_chunks.size());
	maps.Heightmap32F.reserve(1ull);
	for (STPChunk* chk : neighbour_chunks) {
		maps.Biomemap.push_back(chk->getBiomemap());
	}
	maps.Heightmap32F.push_back(current_chunk->getHeightmap());
	maps.HeightmapOffset = this->calcChunkOffset(chunkPos);
	const STPHeightfieldGenerator::STPGeneratorOperation op = STPHeightfieldGenerator::HeightmapGeneration;

	//computing, check success state
	this->generateHeightfield(maps, op);
}

void STPChunkProvider::computeErosion(STPChunkProvider::STPChunkNeighbour& neighbour_chunks) {
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
	this->generateHeightfield(maps, op);
}

#define STORE_EXCEPTION(FUN) try { \
	FUN; \
} \
catch (...) { \
	{ \
		unique_lock<shared_mutex> newExceptionLock(this->ExceptionStorageLock); \
		this->ExceptionStorage.emplace(current_exception()); \
	} \
}

bool STPChunkProvider::prepareNeighbour(vec2 chunkPos, std::function<bool(glm::vec2)>& erosion_reloader, unsigned char rec_depth) {
	//recursive case:
	//define what rec_depth means...
	constexpr static unsigned char BIOMEMAP_PASS = 1u,
		HEIGHTMAP_PASS = 2u;

	{
		STPChunk::STPChunkState expected_state;
		switch (rec_depth) {
		case BIOMEMAP_PASS: expected_state = STPChunk::STPChunkState::Heightmap_Ready;
			break;
		case HEIGHTMAP_PASS: expected_state = STPChunk::STPChunkState::Complete;
			break;
		default:
			break;
		}
		if (STPChunk* center = this->ChunkStorage[chunkPos];
			center != nullptr && center->getChunkState() >= expected_state) {
			//no need to continue if center chunk is available
			//since the center chunk might be used as a neighbour chunk later, we only return bool instead of a pointer
			//after checkChunk() is performed for every chunks, we can grab all pointers and check for occupancy in other functions.
			return true;
		}
	}
	auto biomemap_computer = [this](STPChunk* chunk, vec2 position, vec2 offset) -> void {
		//since biomemap is discrete, we need to round the pixel
		ivec2 rounded_offset = static_cast<ivec2>(glm::round(offset));
		STORE_EXCEPTION(this->generateBiome(chunk->getBiomemap(), ivec3(rounded_offset.x, 0, rounded_offset.y)))
		//computation was successful
		chunk->markChunkState(STPChunk::STPChunkState::Biomemap_Ready);
		chunk->markOccupancy(false);
	};
	auto heightmap_computer = [this](STPChunk* chunk, STPChunkNeighbour neighbours, vec2 position) -> void {
		STORE_EXCEPTION(this->computeHeightmap(chunk, neighbours, position))
		//computation was successful
		chunk->markChunkState(STPChunk::STPChunkState::Heightmap_Ready);
		//unlock all neighbours
		for_each(neighbours.begin(), neighbours.end(), [](auto c) -> void { c->markOccupancy(false); });
	};
	auto erosion_computer = [this](STPChunk* centre, STPChunkNeighbour neighbours) -> void {
		STORE_EXCEPTION(this->computeErosion(neighbours))
		//erosion was successful
		//mark center chunk complete
		centre->markChunkState(STPChunk::STPChunkState::Complete);
		for_each(neighbours.begin(), neighbours.end(), [](auto c) -> void { c->markOccupancy(false); });
	};

	//reminder: central chunk is included in neighbours
	const STPEnvironment::STPChunkSetting& chk_config = this->getChunkSetting();
	const STPChunk::STPChunkPositionCache neighbour_position = this->getNeighbour(chunkPos);

	bool canContinue = true;
	//The first pass: check if all neighbours are ready for some operations
	STPChunkNeighbour neighbour;
	for (vec2 neighbourPos : neighbour_position) {
		//get current neighbour chunk
		STPChunkStorage::STPChunkConstructed res = this->ChunkStorage.construct(neighbourPos, chk_config.MapSize);
		STPChunk* curr_neighbour = res.second;

		if (curr_neighbour->isOccupied()) {
			//occupied means it's currently in used (probably another thread has already started to compute it)
			canContinue = false;
			continue;
		}
		switch (rec_depth) {
		case BIOMEMAP_PASS:
			//container will guaranteed to exists since heightmap pass has already created it
			if (curr_neighbour->getChunkState() == STPChunk::STPChunkState::Empty) {
				curr_neighbour->markOccupancy(true);
				//compute biomemap
				this->kernel_launch_pool.enqueue_void(biomemap_computer, curr_neighbour, neighbourPos, this->calcChunkOffset(neighbourPos));
				//try to compute all biomemap, and when biomemap is computing, we don't need to wait
				canContinue = false;
			}
			break;
		case HEIGHTMAP_PASS:
			//check neighbouring biomemap
			if (!this->prepareNeighbour(neighbourPos, erosion_reloader, rec_depth - 1u)) {
				canContinue = false;
			}
			break;
		default:
			//never gonna happen
			break;
		}

		neighbour.push_back(curr_neighbour);
		//if chunk is found, we can guarantee it's in-used empty or at least biomemap/heightmap complete
	}
	if (!canContinue) {
		//if biomemap/heightmap is computing, we don't need to check for heightmap generation/erosion because some chunks are in use
		return false;
	}

	//The second pass: launch compute on the center with all neighbours
	if (std::any_of(neighbour.begin(), neighbour.end(), [](auto c) -> bool { return c->isOccupied(); })) {
		//if any of the chunk is occupied, we cannot continue
		return false;
	}
	//all chunks are available, lock all neighbours
	for_each(neighbour.begin(), neighbour.end(), [](auto c) -> void { c->markOccupancy(true); });
	//send the list of neighbour chunks to GPU to perform some operations
	switch (rec_depth) {
	case BIOMEMAP_PASS:
		//generate heightmap
		this->kernel_launch_pool.enqueue_void(heightmap_computer, this->ChunkStorage[chunkPos], neighbour, chunkPos);
		break;
	case HEIGHTMAP_PASS:
		//perform erosion on heightmap
		this->kernel_launch_pool.enqueue_void(erosion_computer, this->ChunkStorage[chunkPos], neighbour);
		{
			//trigger a chunk reload as some chunks have been added to render buffer already after neighbours are updated
			const auto neighbour_position = this->getNeighbour(chunkPos);
			for (vec2 position : neighbour_position) {
				erosion_reloader(position);
			}
		}
		break;
	default:
		//never gonna happen
		break;
	}

	//compute has been launched
	return false;
}

bool STPChunkProvider::checkChunk(vec2 chunkPos, std::function<bool(glm::vec2)> reload_callback) {
	//check if there's any exception thrown from previous async compute launch
	bool hasException;
	{
		shared_lock<shared_mutex> checkExceptionLock(this->ExceptionStorageLock);
		hasException = !this->ExceptionStorage.empty();
	}
	if (hasException) {
		unique_lock<shared_mutex> clearExceptionLock(this->ExceptionStorageLock);
		std::stringstream error_message;

		//merge all exception messages
		while (!this->ExceptionStorage.empty()) {
			std::exception_ptr exptr = this->ExceptionStorage.front();
			this->ExceptionStorage.pop();
			try {
				std::rethrow_exception(exptr);
			}
			catch (const std::exception& e) {
				//unfortunately we will lose all exception type information :(
				error_message << e.what() << std::endl;
			}
		}
		//throw the compound exception out
		throw STPException::STPAsyncGenerationError(error_message.str().c_str());
	}

	//recursively preparing neighbours
	if (!this->prepareNeighbour(chunkPos, reload_callback)) {
		//if any neighbour is not ready, and compute has been launched, it will return false
		return false;
	}

	return true;
}

STPChunk* STPChunkProvider::requestChunk(vec2 chunkPos) {
	//after calling checkChunk(), we can guarantee it's not null
	if (STPChunk* chunk = this->ChunkStorage[chunkPos];
		chunk != nullptr) {
		if (!chunk->isOccupied() && chunk->getChunkState() == STPChunk::STPChunkState::Complete) {
			//since we wait for all threads to finish checkChunk(), such that occupancy status will not be changed here
			return chunk;
		}
		return nullptr;
	}
	throw STPException::STPMemoryError("Chunk should have been computed but not found in the storage");
}

const STPEnvironment::STPChunkSetting& STPChunkProvider::getChunkSetting() const {
	return this->ChunkSetting;
}