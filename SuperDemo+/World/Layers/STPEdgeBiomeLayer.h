#pragma once
#ifndef _STP_EDGE_BIOME_LAYER_H_
#define _STP_EDGE_BIOME_LAYER_H_

#include "STPCrossLayer.h"

namespace {

	/**
	 * @brief STPEdgeBiomeLayer adds edge for biome connection, for instance adding beach between ocean and land
	*/
	class STPEdgeBiomeLayer : public STPCrossLayer {
	public:

		STPEdgeBiomeLayer(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer& parent) :
			STPCrossLayer(cache_size, global_seed, salt, parent) {

		}

		Sample sample(const Sample center, const Sample north, const Sample east, const Sample south, const Sample west, Seed) override {
			if (Reg::isOcean(center)) {
				//ocean should be untouched
				return center;
			}

			const bool snowy_area = Reg::getPrecipitationType(center) == Reg::STPPrecipitationType::SNOW;
			//if the centre is land and the surrounding has ocean, turn it into one of the edge biomes
			if (!Reg::applyAll([](const Sample val) -> bool {
				return !Reg::isOcean(val);
			}, north, east, south, west)) {
				//if one of the surrounding is ocean...
				if (snowy_area) {
					//if it's cold...
					return Reg::SnowyBeach.ID;
				}
				if (center == Reg::Mountain.ID) {
					//near the mountain?
					return Reg::StoneShore.ID;
				}

				return Reg::Beach.ID;
			}

			//otherwise everything is untouched
			return center;
		}

	};

}
#endif//_STP_EDGE_BIOME_LAYER_H_