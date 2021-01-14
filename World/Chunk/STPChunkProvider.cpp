#include "STPChunkProvider.h"

using namespace SuperTerrainPlus;

STPChunkProvider::STPChunkProvider(STPSettings::STPConfigurations* settings, STPThreadPool* const shared_threadpool)
	: ChunkSettings(settings->getChunkSettings()), compute_pool(shared_threadpool) {
	//create those workers
	const auto& chunk_settings = settings->getChunkSettings();
	this->heightmap_gen = new STPCompute::STPHeightfieldGenerator(&settings->getHeightfieldSettings(), &settings->getSimplexNoiseSettings());
	this->formatter = new STPCompute::STPImageConverter(make_uint2(chunk_settings.MapSize.x, chunk_settings.MapSize.y));
}

STPChunkProvider::~STPChunkProvider() {
	//delete those generators and workers
	delete this->heightmap_gen;
	delete this->formatter;
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

	//computing
	bool result = this->heightmap_gen->generateHeightfieldCUDA(current_chunk->getHeightmap(), current_chunk->getNormalmap(),
		//first convert chunk world position to relative chunk position, then multiply by the map size, such that the generated map will be seamless
		make_float3(
			//we substract the mapsize by 1 for the offset
			//such that the first row of pixels in the next chunk will be the same as the last row in the previous
			//to achieve seamless experience :)
			static_cast<float>(this->ChunkSettings.MapSize.x - 1u) * chunkPos.x / (static_cast<float>(this->ChunkSettings.ChunkSize.x) * this->ChunkSettings.ChunkScaling) + this->ChunkSettings.MapOffset.x,
			this->ChunkSettings.MapOffset.y,
			static_cast<float>(this->ChunkSettings.MapSize.y - 1u) * chunkPos.y / (static_cast<float>(this->ChunkSettings.ChunkSize.y) * this->ChunkSettings.ChunkScaling) + this->ChunkSettings.MapOffset.z));

	if (result) {//computation was successful
		std::atomic_store_explicit<STPChunk::STPChunkState>(
			&current_chunk->State, STPChunk::STPChunkState::Erosion_Ready, std::memory_order::memory_order_relaxed);

		//now converting image format to 16bit
		result &= this->formatChunk(current_chunk);
	}

	return result;
}

bool STPChunkProvider::formatChunk(STPChunk* const current_chunk) {
	if (std::atomic_load_explicit<STPChunk::STPChunkState>(
		&current_chunk->State, std::memory_order::memory_order_relaxed) != STPChunk::STPChunkState::Complete) {
		//A static warpper to call the class function async
		auto conversion_warpper = [converter = this->formatter](const float* original, unsigned short* output, int channel) -> bool {
			//The function only read the mapsize from the class, so we can safely launch the converter in multithread
			return converter->floatToshortCUDA(original, output, channel);
		};

		std::future<bool> conversion_workers[2];
		bool status = true;
		status &= this->formatter->floatToshortCUDA(current_chunk->TerrainMaps[0], current_chunk->TerrainMaps_cache[0], 1);
		status &= this->formatter->floatToshortCUDA(current_chunk->TerrainMaps[1], current_chunk->TerrainMaps_cache[1], 4);

		//update the status
		std::atomic_store_explicit<STPChunk::STPChunkState>(
			&current_chunk->State, STPChunk::STPChunkState::Complete, std::memory_order::memory_order_relaxed);
		std::atomic_store_explicit<bool>(&current_chunk->inUsed, false, std::memory_order::memory_order_relaxed);
		return status;
	}

	//no computation was dispatched
	return false;
}

STPChunkProvider::STPChunkLoaded STPChunkProvider::requestChunk(STPChunkStorage* source, vec2 chunkPos) {
	//check if chunk exists
	//lock the thread in shared state when writing
	STPChunk* storage_unit = nullptr;
	{
		std::shared_lock<std::shared_mutex> read_lock(this->chunk_storage_locker);
		storage_unit = source->getChunk(chunkPos);
	}

	if (storage_unit == nullptr) {
		//chunk not found
		//first we create an empty chunk, with default initial status

		//map computatin launch warpper
		//since we are only using the class object as function pointer, we are not modifying it
		auto computer_warpper = [this](STPChunk* const current_chunk, vec2 chunkPos) -> bool {
			return this->computeChunk(current_chunk, chunkPos);
		};

		//a new chunk
		STPChunk* const current_chunk = new STPChunk(this->ChunkSettings.MapSize, true);
		std::atomic_store_explicit<bool>(&current_chunk->inUsed, true, std::memory_order::memory_order_relaxed);
		//lock the thread while writing into the data structure
		{
			std::unique_lock<std::shared_mutex> write_lock(this->chunk_storage_locker);//it will handle exception for us
			source->addChunk(chunkPos, current_chunk);//insertion is guarateed since we know chunk not found
		}

		//then dispatch compute in another thread, the results will be copied to the new chunk directly
		//we are only passing the pointer to the chunk (not the entire container), and each thread only deals with one chunk, so shared_read lock is not requried
		this->compute_pool->enqueue_void(computer_warpper, current_chunk, chunkPos);

		//the chunk is not found and it's now being computed
		return std::make_pair(STPChunkReadyStatus::Not_Found, nullptr);
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
		if (!std::atomic_load_explicit<bool>(&storage_unit->inUsed, std::memory_order::memory_order_relaxed) &&
			std::atomic_load_explicit<STPChunk::STPChunkState>(
				&storage_unit->State, std::memory_order::memory_order_relaxed) == STPChunk::STPChunkState::Complete) {
			//chunk is ready, we can return
			return std::make_pair(STPChunkReadyStatus::Complete, storage_unit);
		}

		//chunk is in used by other threads
		return std::make_pair(STPChunkReadyStatus::In_Used, nullptr);
	}
}

const STPSettings::STPChunkSettings* STPChunkProvider::getChunkSettings() {
	return &(this->ChunkSettings);
}

bool STPChunkProvider::setHeightfieldErosionIteration(unsigned int iteration) {
	return this->heightmap_gen->setErosionIterationCUDA(iteration);
}