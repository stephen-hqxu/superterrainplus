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

		STPRiverErodeLayer(const size_t cache_size, const STPSeed_t global_seed, const STPSeed_t salt, STPLayer& parent) :
			STPCrossLayer(cache_size, global_seed, salt, parent) {
			//parent: noise to river layer
		}

		STPSample_t sample(const STPSample_t center, const STPSample_t north, const STPSample_t east,
			const STPSample_t south, const STPSample_t west, STPSeed_t) override {
			//centre is river, return a river if any of the sample is a river
			if (!Reg::applyAll([](const STPSample_t val) -> bool {
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