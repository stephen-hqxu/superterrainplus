#include "STPBiomeFactory.h"

using namespace SuperTerrainPlus::STPBiome;

STPBiomeFactory::STPBiomeFactory(uvec3 dimension) : BiomeDimension(dimension) {

}

STPBiomeFactory::STPBiomeFactory(uvec2 dimension) : BiomeDimension(dimension.x, 1u, dimension.y) {

}

STPBiomeFactory::~STPBiomeFactory() {

}

const Sample* STPBiomeFactory::generate(STPLayer* const chain, ivec3 offset) {
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

void STPBiomeFactory::dump(Sample* map) {
	delete[] map;
}