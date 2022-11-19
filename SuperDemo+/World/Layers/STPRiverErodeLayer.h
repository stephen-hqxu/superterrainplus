#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "STPCrossLayer.h"
#include "../Biomes/STPBiomeRegistry.h"

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPRiverErodeLayer erodes the river path to make it larger
	*/
	class STPRiverErodeLayer : public STPCrossLayer {
	public:

		STPRiverErodeLayer(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer* const parent) :
			STPCrossLayer(cache_size, global_seed, salt, parent) {
			//parent: noise to river layer
		}

		Sample sample(const Sample center, const Sample north, const Sample east, const Sample south, const Sample west, Seed) override {
			//centre is river, return a river if any of the sample is a river
			if (!STPBiomeRegistry::applyAll([](const Sample val) -> bool {
				return val != STPBiomeRegistry::River.ID;
				}, north, east, south, west)) {
				return STPBiomeRegistry::River.ID;
			}
			
			//otherwise don't touch
			return center;
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_