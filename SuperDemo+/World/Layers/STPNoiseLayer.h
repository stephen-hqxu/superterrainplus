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
	 * @brief STPNoiseLayer generates random value on non-ocean biomes, this will be mainly used for river network generation
	*/
	class STPNoiseLayer : public SuperTerrainPlus::STPDiversity::STPLayer {
	public:

		STPNoiseLayer(Seed global_seed, Seed salt, STPLayer* parent) : STPLayer(global_seed, salt, parent) {
			//noise layer will overwrite previous interpretation, this is a new chain of layers
		}

		Sample sample(int x, int y, int z) override {
			//reset local seed
			const Seed local_seed = this->genLocalSeed(x, z);
			//get the local generator
			const STPLayer::STPLocalRNG rng = this->getRNG(local_seed);

			//value from the previous layer
			const Sample val = this->getAscendant()->retrieve(x, y, z);
			//leaving ocean untouched, given a random noise value for the river generation layer
			return STPBiomeRegistry::isShallowOcean(val) ? val : rng.nextVal(29999) + 2;
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_