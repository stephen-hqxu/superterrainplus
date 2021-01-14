#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "STPCrossLayer.h"
#include "../STPBiome_def.h"

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPBiome::Seed;
	using SuperTerrainPlus::STPBiome::Sample;

	/**
	 * @brief STPEdgeBiomeLayer adds edge for biome connection, for instance adding beach between ocean and land
	*/
	class STPEdgeBiomeLayer : public STPCrossLayer {
	public:

		STPEdgeBiomeLayer(Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(global_seed, salt, parent) {

		}

		Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed local_seed) {
			if (STPBiomeRegistry::isOcean(center)) {
				//ocean should be untouched
				return center;
			}

			const bool snowy_area = STPBiomeRegistry::getPrecipitationType(center) == STPBiomeRegistry::STPPrecipitationType::SNOW;
			//if the center is land and the surrounding has ocean, turn it into one of the edge biomes
			if (!STPBiomeRegistry::applyAll([](Sample val) -> bool {
				return !STPBiomeRegistry::isOcean(val);
			}, north, east, south, west)) {
				//if one of the surrounding is ocean...
				if (snowy_area) {
					//if it's cold...
					return STPBiomeRegistry::SNOWY_BEACH.getID();
				}
				if (center == STPBiomeRegistry::MOUNTAIN.getID()) {
					//near the mountain?
					return STPBiomeRegistry::STONE_SHORE.getID();
				}

				return STPBiomeRegistry::BEACH.getID();
			}

			//otherwise everything is untouched
			return center;
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_