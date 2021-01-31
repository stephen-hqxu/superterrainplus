#include "STPBiomeFactory.h"

using glm::uvec2;
using glm::uvec3;
using glm::ivec3;

using namespace SuperTerrainPlus::STPBiome;

STPLayer* STPBiomeFactory::STPBiomeAllocator::allocate(size_t size, STPManufacturer manufacturer) {
	//size will always be one btw, memory is allocated and object will be constructed right away
	return manufacturer();
}

void STPBiomeFactory::STPBiomeAllocator::deallocate(size_t size, STPLayer* layer) {
	STPLayer::destroy(layer);
}

STPBiomeFactory::STPBiomeFactory(uvec3 dimension) : BiomeDimension(dimension) {
	
}

STPBiomeFactory::STPBiomeFactory(glm::uvec3 dimension, STPManufacturer manufacturer) : STPBiomeFactory(dimension) {
	//assign allocator
	this->manufacturer = manufacturer;
}

STPBiomeFactory::STPBiomeFactory(uvec2 dimension) : BiomeDimension(dimension.x, 1u, dimension.y) {

}

STPBiomeFactory::STPBiomeFactory(glm::uvec2 dimension, STPManufacturer manufacturer)
	: STPBiomeFactory(uvec3(dimension.x, 1u, dimension.y), manufacturer) {

}

STPBiomeFactory::~STPBiomeFactory() {
	//make sure the thread stopped before deleting factory
	//stop all waiting workers and waiting for current worker to finish.
}

size_t STPBiomeFactory::size() const {
	size_t count;
	{
		std::shared_lock<std::shared_mutex> read_lock(this->cache_lock);
		count = this->layer_cache.size();
	}
	return count;
}

const Sample* STPBiomeFactory::generate(STPLayer* const chain, ivec3 offset) const {
	Sample* const map = new Sample[this->BiomeDimension.x * this->BiomeDimension.y * this->BiomeDimension.z];

	//loop through and generate the biome map
	//why not using CUDA and do it in parallel? Because the biome layers are cached, tested and parallel performance is a piece of shit
	if (this->BiomeDimension.y == 1u) {
		//it's a 2D biome
		//to avoid making useless computation
		for (unsigned int x = 0u; x < this->BiomeDimension.x; x++) {
			for (unsigned int z = 0u; z < this->BiomeDimension.z; z++) {
				//calculate the map index
				const unsigned int index = x + z * this->BiomeDimension.x;
				//get the biome at thie coordinate
				map[index] = chain->sample(static_cast<int>(x) + offset.x, 0, static_cast<int>(z) + offset.z);
			}
		}
	}
	else {
		//it's a 3D biome
		for (unsigned int x = 0u; x < this->BiomeDimension.x; x++) {
			for (unsigned int y = 0u; y < this->BiomeDimension.y; y++) {
				for (unsigned int z = 0u; z < this->BiomeDimension.z; z++) {
					//calculate the map index
					const unsigned int index = x + y * this->BiomeDimension.x + z * (this->BiomeDimension.x * this->BiomeDimension.y);
					//get the biome at thie coordinate
					map[index] = chain->sample(static_cast<int>(x) + offset.x, static_cast<int>(y) + offset.y, static_cast<int>(z) + offset.z);
				}
			}
		}
	}

	return const_cast<const Sample*>(map);
}

const Sample* STPBiomeFactory::generate(glm::ivec3 offset) {
	//it's a thread safe function
	if (this->manufacturer == nullptr) {
		//prevent the thread from deadlocking if there is no cache
		throw std::runtime_error("No cache has associated with the biome factory.");
	}

	STPLayer* layer = nullptr;
	{
		//try to grab the lock
		std::unique_lock<std::shared_mutex> lock(this->cache_lock);
		//get memory
		layer = this->layer_cache.allocate(1ull, this->manufacturer);
		
	}
	
	//start the generation
	const Sample* sample = this->generate(layer, offset);

	{
		//return the cache back
		std::unique_lock<std::shared_mutex> lock(this->cache_lock);
		this->layer_cache.deallocate(1ull, layer);
	}

	return sample;
}

void STPBiomeFactory::dump(Sample* map) {
	delete[] map;
}