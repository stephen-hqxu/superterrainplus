#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "STPCrossLayer.h"
#include "../Biomes/STPBiomeRegistry.h"

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPEdgeBiomeLayer adds edge for biome connection, for instance adding beach between ocean and land
	*/
	class STPEdgeBiomeLayer : public STPCrossLayer {
	public:

		STPEdgeBiomeLayer(size_t cache_size, Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(cache_size, global_seed, salt, parent) {

		}

		Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed) {
			if (STPBiomeRegistry::isOcean(center)) {
				//ocean should be untouched
				return center;
			}

			const bool snowy_area = STPBiomeRegistry::getPrecipitationType(center) == STPBiomeRegistry::STPPrecipitationType::SNOW;
			//if the centre is land and the surrounding has ocean, turn it into one of the edge biomes
			if (!STPBiomeRegistry::applyAll([](Sample val) -> bool {
				return !STPBiomeRegistry::isOcean(val);
			}, north, east, south, west)) {
				//if one of the surrounding is ocean...
				if (snowy_area) {
					//if it's cold...
					return STPBiomeRegistry::SnowyBeach.ID;
				}
				if (center == STPBiomeRegistry::Mountain.ID) {
					//near the mountain?
					return STPBiomeRegistry::StoneShore.ID;
				}

				return STPBiomeRegistry::Beach.ID;
			}

			//otherwise everything is untouched
			return center;
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_