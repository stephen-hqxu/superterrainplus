#include "STPBiomeFactory.h"

using glm::uvec2;
using glm::uvec3;
using glm::ivec3;

using namespace SuperTerrainPlus::STPBiome;

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

const Sample* STPBiomeFactory::generate(STPLayerManager* chain, ivec3 offset) const {
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
				map[index] = chain->start()->sample(static_cast<int>(x) + offset.x, 0, static_cast<int>(z) + offset.z);
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
					map[index] = chain->start()->sample(static_cast<int>(x) + offset.x, static_cast<int>(y) + offset.y, static_cast<int>(z) + offset.z);
				}
			}
		}
	}

	return const_cast<const Sample*>(map);
}