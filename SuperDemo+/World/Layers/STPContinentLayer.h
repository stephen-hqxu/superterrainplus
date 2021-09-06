#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include <SuperTerrain+/World/Diversity/STPLayer.h>
#include "../Biomes/STPBiomeRegistry.h"

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPContinentLayer is the first layer of the biome generation chain, it generates land and ocean section
	*/
	class STPContinentLayer : public SuperTerrainPlus::STPDiversity::STPLayer {
	public:

		STPContinentLayer(Seed global_seed, Seed salt) : STPLayer(global_seed, salt) {

		}

		Sample sample(int x, int y, int z) override {
			//set local seed
			const Seed local_seed = this->genLocalSeed(x, z);
			//get the rng for this coordinate
			const STPLayer::STPLocalRNG rng = this->getRNG(local_seed);

			//we give 1/10 chance for land
			return rng.nextVal(10u) == 0 ? STPBiomeRegistry::PLAINS.getID() : STPBiomeRegistry::OCEAN.getID();
		}
	};
}
#endif//_STP_LAYERS_ALL_HPP_