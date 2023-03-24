#pragma once
#ifndef _STP_NOISE_TO_RIVER_LAYER_H_
#define _STP_NOISE_TO_RIVER_LAYER_H_

#include "STPCrossLayer.h"

namespace {

	/**
	 * @brief STPNoiseToRiverLayer converts to land noise layer to river by sampling the same noise value
	*/
	class STPNoiseToRiverLayer : public STPCrossLayer {
	private:

		inline static Sample filterRiver(Sample val) {
			//giving 1/2 chance of having a river
			return val >= 2 ? 2 + (val & 1) : val;
		}

	public:

		STPNoiseToRiverLayer(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer& parent) :
			STPCrossLayer(cache_size, global_seed, salt, parent) {

		}

		Sample sample(const Sample center, const Sample north, const Sample east, const Sample south, const Sample west, Seed) override {
			//filter the river
			//basically it's an edge detector
			const Sample i = STPNoiseToRiverLayer::filterRiver(center);
			return i == STPNoiseToRiverLayer::filterRiver(north) && i == STPNoiseToRiverLayer::filterRiver(east)
				&& i == STPNoiseToRiverLayer::filterRiver(south) && i == STPNoiseToRiverLayer::filterRiver(west)
				? 0xFFFFu : Reg::River.ID;
		}

	};

}
#endif//_STP_NOISE_TO_RIVER_LAYER_H_