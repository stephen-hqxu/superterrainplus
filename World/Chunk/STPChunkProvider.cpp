#include "STPChunkProvider.h"

using glm::vec2;

using namespace SuperTerrainPlus;

STPChunkProvider::STPChunkProvider(STPSettings::STPConfigurations* settings)
	: ChunkSettings(settings->getChunkSettings())
	, heightmap_gen(&settings->getSimplexNoiseSettings()) {

}

bool STPChunkProvider::computeChunk(STPChunk* const current_chunk, vec2 chunkPos) {
	//determine whethwe we need to recompute the terrain or not
	//Simplified using boolean algebra
	/*bool recompute = false;
	if (overwrite) {
		recompute = true;
	}
	else if (!overwrite && current_chunk->Chunk_Completed) {
		recompute = false;
	}
	else if (!current_chunk->Chunk_Completed) {
		recompute = true;
	}*/
	using namespace STPCompute;
	//first convert chunk world position to relative chunk position, then multiply by the map size, such that the generated map will be seamless
	const float3 offset = make_float3(
		//we substract the mapsize by 1 for the offset
		//such that the first row of pixels in the next chunk will be the same as the last row in the previous
		//to achieve seamless experience :)
		static_cast<float>(this->ChunkSettings.MapSize.x - 1u) * chunkPos.x / (static_cast<float>(this->ChunkSettings.ChunkSize.x) * this->ChunkSettings.ChunkScaling) + this->ChunkSettings.MapOffset.x,
		this->ChunkSettings.MapOffset.y,
		static_cast<float>(this->ChunkSettings.MapSize.y - 1u) * chunkPos.y / (static_cast<float>(this->ChunkSettings.ChunkSize.y) * this->ChunkSettings.ChunkScaling) + this->ChunkSettings.MapOffset.z
	);

	STPHeightfieldGenerator::STPMapStorage maps;
	maps.Heightmap32F.push_back(current_chunk->getRawMap(STPChunk::STPMapType::Heightmap));
	maps.HeightmapOffset = offset;
	maps.Normalmap32F = current_chunk->getRawMap(STPChunk::STPMapType::Normalmap);
	const STPHeightfieldGenerator::STPGeneratorOperation op =
		STPHeightfieldGenerator::HeightmapGeneration | STPHeightfieldGenerator::Erosion | 
		STPHeightfieldGenerator::NormalmapGeneration | STPHeightfieldGenerator::Format;
	maps.FormatHint = STPHeightfieldGenerator::FormatHeightmap | STPHeightfieldGenerator::FormatNormalmap;
	maps.Heightmap16UI = current_chunk->getCacheMap(STPChunk::STPMapType::Heightmap);
	maps.Normalmap16UI = current_chunk->getCacheMap(STPChunk::STPMapType::Normalmap);

	//computing
	bool result = this->heightmap_gen.generateHeightfieldCUDA(maps, op);

	if (result) {//computation was successful
		current_chunk->markChunkState(STPChunk::STPChunkState::Complete);
		current_chunk->markOccupancy(false);

		//now converting image format to 16bit
		//result &= this->formatChunk(current_chunk);
	}

	return result;
}

STPChunkProvider::STPChunkLoaded STPChunkProvider::requestChunk(STPChunkStorage& source, vec2 chunkPos) {
	//check if chunk exists
	//lock the thread in shared state when writing
	STPChunk* storage_unit = nullptr;
	storage_unit = source.getChunk(chunkPos);

	if (storage_unit == nullptr) {
		//chunk not found
		//first we create an empty chunk, with default initial status

		//a new chunk
		STPChunk* const current_chunk = new STPChunk(this->ChunkSettings.MapSize, true);
		current_chunk->markOccupancy(true);
		//lock the thread while writing into the data structure
		source.addChunk(chunkPos, current_chunk);//insertion is guarateed since we know chunk not found

		//then dispatch compute in another thread, the results will be copied to the new chunk directly
		//we are only passing the pointer to the chunk (not the entire container), and each thread only deals with one chunk, so shared_read lock is not requried
		if (this->computeChunk(current_chunk, chunkPos)) {
			//computed, chunk can be used
			return std::make_pair(true, current_chunk);
		}

		//the chunk is not found and it cannot be computed
		throw std::runtime_error("Chunk generation failed");
	}
	else {
		//chunk found
		//check if it has been completed
		//if (!current_chunk->Chunk_Completed) {
		//	//computation in progress (on other threads)
		//	//result will be copied back to this chunk by that thread once finished
		//	return false;
		//}
		//if (current_chunk->Chunk_Completed && !current_chunk->Memory_Updated) {
		//	//we only need to update the cache if the map has been recomputed
		//	//such that MapConverter() will only be called by MapComputer()
		//	return false;
		//}
		//simplified with boolean algebra and invert it to true condition
		if (!storage_unit->isOccupied() && storage_unit->getChunkState() == STPChunk::STPChunkState::Complete) {
			//chunk is ready, we can return
			return std::make_pair(true, storage_unit);
		}

		//chunk is in used by other threads
		return std::make_pair(false, nullptr);
	}
}

const STPSettings::STPChunkSettings* STPChunkProvider::getChunkSettings() const {
	return &(this->ChunkSettings);
}

bool STPChunkProvider::setHeightfieldErosionIteration(unsigned int iteration) {
	return this->heightmap_gen.setErosionIterationCUDA(iteration);
}