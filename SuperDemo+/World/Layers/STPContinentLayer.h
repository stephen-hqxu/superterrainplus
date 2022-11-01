#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include <SuperTerrain+/World/Diversity/STPLayer.h>
#include "../Biomes/STPBiomeRegistry.h"

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPContinentLayer is the first layer of the biome generation chain, it generates land and ocean section
	*/
	class STPContinentLayer : public SuperTerrainPlus::STPDiversity::STPLayer {
	public:

		STPContinentLayer(size_t cache_size, Seed global_seed, Seed salt) : STPLayer(cache_size, global_seed, salt) {

		}

		Sample sample(int x, int, int z) override {
			//set local seed
			const Seed local_seed = this->genLocalSeed(x, z);
			//get the RNG for this coordinate
			const STPLayer::STPLocalRNG rng = this->getRNG(local_seed);

			//we give 1/10 chance for land
			return rng.nextVal(10u) == 0 ? STPBiomeRegistry::Plains.ID : STPBiomeRegistry::Ocean.ID;
		}
	};
}
#endif//_STP_LAYERS_ALL_HPP_