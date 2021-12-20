#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "STPCrossLayer.h"
#include "../Biomes/STPBiomeRegistry.h"

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPDeepOceanLayer adds deep ocean if it's surrounded by shallow ocean and not next to land
	*/
	class STPDeepOceanLayer : public STPCrossLayer {
	public:

		STPDeepOceanLayer(Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(global_seed, salt, parent) {

		}

		Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed) override {
			//if the center is not shallow or it's not even ocean, we can't change anything
			if (!STPBiomeRegistry::isShallowOcean(center)) {
				return center;
			}

			//test if we are surrounded by shallow ocean
			Sample i = 0;
			if (STPBiomeRegistry::isShallowOcean(north)) {
				i++;
			}
			if (STPBiomeRegistry::isShallowOcean(east)) {
				i++;
			}
			if (STPBiomeRegistry::isShallowOcean(south)) {
				i++;
			}
			if (STPBiomeRegistry::isShallowOcean(west)) {
				i++;
			}
			
			//if we are surrounded, generate their relative deep ocean
			if (i > 3) {
				if (center == STPBiomeRegistry::OCEAN.getID()) {
					return STPBiomeRegistry::DEEP_OCEAN.getID();
				}
				if (center == STPBiomeRegistry::WARM_OCEAN.getID()) {
					return STPBiomeRegistry::DEEP_WARM_OCEAN.getID();
				}
				if (center == STPBiomeRegistry::LUKEWARM_OCEAN.getID()) {
					return STPBiomeRegistry::DEEP_LUKEWARM_OCEAN.getID();
				}
				if (center == STPBiomeRegistry::COLD_OCEAN.getID()) {
					return STPBiomeRegistry::DEEP_COLD_OCEAN.getID();
				}
				if (center == STPBiomeRegistry::FROZEN_OCEAN.getID()) {
					return STPBiomeRegistry::DEEP_FROZEN_OCEAN.getID();
				}

			}

			return center;
		}
	};
}
#endif//_STP_LAYERS_ALL_HPP_