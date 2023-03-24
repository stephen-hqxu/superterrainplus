#pragma once
#ifndef _STP_RIVER_ERODE_LAYER_H_
#define _STP_RIVER_ERODE_LAYER_H_

#include "STPCrossLayer.h"

namespace {

	/**
	 * @brief STPRiverErodeLayer erodes the river path to make it larger
	*/
	class STPRiverErodeLayer : public STPCrossLayer {
	public:

		STPRiverErodeLayer(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer& parent) :
			STPCrossLayer(cache_size, global_seed, salt, parent) {
			//parent: noise to river layer
		}

		Sample sample(const Sample center, const Sample north, const Sample east, const Sample south, const Sample west, Seed) override {
			//centre is river, return a river if any of the sample is a river
			if (!Reg::applyAll([](const Sample val) -> bool {
				return val != Reg::River.ID;
				}, north, east, south, west)) {
				return Reg::River.ID;
			}
			
			//otherwise don't touch
			return center;
		}

	};

}
#endif//_STP_RIVER_ERODE_LAYER_H_