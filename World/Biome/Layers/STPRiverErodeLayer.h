#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "STPCrossLayer.h"
#include "../Biomes/STPBiomeRegistry.h"

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPBiome::Seed;
	using SuperTerrainPlus::STPBiome::Sample;

	/**
	 * @brief STPRiverErodeLayer erodes the river path to make it larger
	*/
	class STPRiverErodeLayer : public STPCrossLayer {
	public:

		STPRiverErodeLayer(Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(global_seed, salt, parent) {
			//parent: noise to river layer
		}

		Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed local_seed) override {
			//center is river, return a river if any of the sample is a river
			//de morgan's law
			if (!STPBiomeRegistry::applyAll([](Sample val) -> bool {
				return val != STPBiomeRegistry::RIVER.getID();
				}, north, east, south, west)) {
				return STPBiomeRegistry::RIVER.getID();
			}
			
			//otherwise don't touch
			return center;
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_