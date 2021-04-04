#include "STPChunkProvider.h"

using glm::vec2;

using std::make_pair;

using namespace SuperTerrainPlus;

STPChunkProvider::STPChunkProvider(STPSettings::STPConfigurations* settings)
	: ChunkSettings(settings->getChunkSettings())
	, heightmap_gen(&settings->getSimplexNoiseSettings(), &this->ChunkSettings) {

}

float3 STPChunkProvider::calcChunkOffset(vec2 chunkPos) const {
	//first convert chunk world position to relative chunk position, then multiply by the map size, such that the generated map will be seamless
	return make_float3(
		//we substract the mapsize by 1 for the offset
		//such that the first row of pixels in the next chunk will be the same as the last row in the previous
		//to achieve seamless experience :)
		static_cast<float>(this->ChunkSettings.MapSize.x - 1u) * chunkPos.x / (static_cast<float>(this->ChunkSettings.ChunkSize.x) * this->ChunkSettings.ChunkScaling) + this->ChunkSettings.MapOffset.x,
		this->ChunkSettings.MapOffset.y,
		static_cast<float>(this->ChunkSettings.MapSize.y - 1u) * chunkPos.y / (static_cast<float>(this->ChunkSettings.ChunkSize.y) * this->ChunkSettings.ChunkScaling) + this->ChunkSettings.MapOffset.z
	);
}

bool STPChunkProvider::computeHeightmap(STPChunk* const current_chunk, vec2 chunkPos) {
	using namespace STPCompute;

	STPHeightfieldGenerator::STPMapStorage maps;
	maps.Heightmap32F.push_back(current_chunk->getRawMap(STPChunk::STPMapType::Heightmap));
	maps.HeightmapOffset = this->calcChunkOffset(chunkPos);
	const STPHeightfieldGenerator::STPGeneratorOperation op = STPHeightfieldGenerator::HeightmapGeneration;

	//computing
	bool result = this->heightmap_gen.generateHeightfieldCUDA(maps, op);

	if (result) {//computation was successful
		current_chunk->markChunkState(STPChunk::STPChunkState::Heightmap_Ready);
		current_chunk->markOccupancy(false);
	}

	return result;
}

bool STPChunkProvider::computeErosion(STPChunk* const current_chunk, std::list<STPChunk*>& neighbour_chunks) {
	using namespace STPCompute;

	STPHeightfieldGenerator::STPMapStorage maps;
	for (STPChunk* chk : neighbour_chunks) {
		maps.Heightmap32F.push_back(chk->getRawMap(STPChunk::STPMapType::Heightmap));
	}
	maps.Normalmap32F = current_chunk->getRawMap(STPChunk::STPMapType::Normalmap);
	const STPHeightfieldGenerator::STPGeneratorOperation op =
		STPHeightfieldGenerator::Erosion |
		STPHeightfieldGenerator::NormalmapGeneration | STPHeightfieldGenerator::Format;
	maps.FormatHint = STPHeightfieldGenerator::FormatHeightmap | STPHeightfieldGenerator::FormatNormalmap;
	maps.Heightmap16UI = current_chunk->getCacheMap(STPChunk::STPMapType::Heightmap);
	maps.Normalmap16UI = current_chunk->getCacheMap(STPChunk::STPMapType::Normalmap);

	//computing
	bool result = this->heightmap_gen.generateHeightfieldCUDA(maps, op);

	if (result) {//computation was successful
		//unlock all neighbours
		for (STPChunk* chk : neighbour_chunks) {
			chk->markOccupancy(false);
		}
		//mark center chunk complete
		current_chunk->markChunkState(STPChunk::STPChunkState::Complete);
	}

	return result;
}

bool STPChunkProvider::checkChunk(STPChunkStorage& source, vec2 chunkPos) {
	//EXPERIMENTAL FEATURE
	//TODO: This looks excessively complicated, simplification is needed
	STPChunk* center = source.getChunk(chunkPos);
	if (center != nullptr && center->getChunkState() == STPChunk::STPChunkState::Complete) {
		//no need to continue if center chunk is available
		//since the center chunk might be used as a neighbour chunk later, we only return bool instead of a pointer
		//after checkChunk() is performed for every chunks, we can grab all pointers and check for occupancy in other functions.
		return true;
	}
	//reminder: central chunk is included in neighbours
	const STPSettings::STPChunkSettings* chk_config = this->getChunkSettings();
	const STPChunk::STPChunkPosCache neighbours = STPChunk::getRegion(chunkPos, chk_config->ChunkSize, chk_config->FreeSlipChunk, chk_config->ChunkScaling);
	bool canContinue = true;
	
	//The first pass: check if all neighbours are heightmap-complete
	std::list<STPChunk*> neighbour;
	for (vec2 neighbourPos : neighbours) {
		//get current neighbour chunk
		STPChunkStorage::STPChunkConstructed res = source.constructChunk(neighbourPos, chk_config->MapSize);
		STPChunk* curr_neighbour = res.second;
		if (res.first) {
			//neighbour doesn't exist and has been added
			if (!this->computeHeightmap(curr_neighbour, neighbourPos)) {
				//if compute is OK -> canContinue = true, otherwise false
				throw std::runtime_error("Heightmap computation failed");
			}
			//if continued, keep checking rest of the chunks
		}
		neighbour.push_back(curr_neighbour);
		//if chunk is found, we can guarantee it's in-used empty or at least heightmap complete
	}
	if (!canContinue) {
		return canContinue;
	}

	//The second pass: launch full compute
	{
		std::unique_lock<std::mutex> lock(this->neighbour_lock);
		for (STPChunk* curr_neighbour : neighbour) {
			if (curr_neighbour->isOccupied()) {
				canContinue = false;
				break;
			}
		}
		if (canContinue) {
			for (STPChunk* curr_neighbour : neighbour) {
				curr_neighbour->markOccupancy(true);
			}
			//send the list of neighbour chunks to GPU to perform free-slip hydraulic erosion
			if (!this->computeErosion(source.getChunk(chunkPos), neighbour)) {
				throw std::runtime_error("Hydraulic erosion simulation failed");
			}
		}
	}

	return canContinue;
}

STPChunk* STPChunkProvider::requestChunk(STPChunkStorage& source, vec2 chunkPos) {
	//EXPERIMENTAL FEATURE
	//after calling checkChunk(), we can guarantee it's not null
	STPChunk* chunk = source.getChunk(chunkPos);
	if (chunk != nullptr) {
		if (!chunk->isOccupied()) {
			//since we wait for all threads to finish checkChunk(), such that occupancy status will not be changed here
			return chunk;
		}
		return nullptr;
	}
	throw std::runtime_error("Chunk chunk should have been computed but not found");
}

const STPSettings::STPChunkSettings* STPChunkProvider::getChunkSettings() const {
	return &(this->ChunkSettings);
}

bool STPChunkProvider::setHeightfieldErosionIteration(unsigned int iteration) {
	return this->heightmap_gen.setErosionIterationCUDA(iteration);
}
