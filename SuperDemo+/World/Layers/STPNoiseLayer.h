#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include <SuperTerrain+/World/Diversity/STPLayer.h>
#include "../Biomes/STPBiomeRegistry.h"

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPNoiseLayer generates random value on non-ocean biomes, this will be mainly used for river network generation
	*/
	class STPNoiseLayer : public SuperTerrainPlus::STPDiversity::STPLayer {
	public:

		STPNoiseLayer(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer* const parent) :
			STPLayer(cache_size, global_seed, salt, parent) {
			//noise layer will overwrite previous interpretation, this is a new chain of layers
		}

		Sample sample(const int x, const int y, const int z) override {
			//get the local generator
			const STPLayer::STPLocalSampler rng = this->createLocalSampler(x, z);

			//value from the previous layer
			const Sample val = this->getAscendant()->retrieve(x, y, z);
			//leaving ocean untouched, given a random noise value for the river generation layer
			return STPBiomeRegistry::isShallowOcean(val) ? val : rng.nextValue(29999) + 2;
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_