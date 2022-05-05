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

		STPRiverErodeLayer(Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(global_seed, salt, parent) {
			//parent: noise to river layer
		}

		Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed) override {
			//centre is river, return a river if any of the sample is a river
			if (!STPBiomeRegistry::applyAll([](Sample val) -> bool {
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