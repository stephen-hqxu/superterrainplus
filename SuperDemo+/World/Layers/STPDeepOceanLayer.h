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

		STPDeepOceanLayer(size_t cache_size, Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(cache_size, global_seed, salt, parent) {

		}

		Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed) override {
			//if the centre is not shallow or it's not even ocean, we can't change anything
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
				if (center == STPBiomeRegistry::Ocean.ID) {
					return STPBiomeRegistry::DeepOcean.ID;
				}
				if (center == STPBiomeRegistry::WarmOcean.ID) {
					return STPBiomeRegistry::DeepWarmOcean.ID;
				}
				if (center == STPBiomeRegistry::LukewarmOcean.ID) {
					return STPBiomeRegistry::DeepLukewarmOcean.ID;
				}
				if (center == STPBiomeRegistry::ColdOcean.ID) {
					return STPBiomeRegistry::DeepColdOcean.ID;
				}
				if (center == STPBiomeRegistry::FrozenOcean.ID) {
					return STPBiomeRegistry::DeepFrozenOcean.ID;
				}

			}

			return center;
		}
	};
}
#endif//_STP_LAYERS_ALL_HPP_