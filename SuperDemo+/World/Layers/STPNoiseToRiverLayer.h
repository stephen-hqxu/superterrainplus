#pragma once
#ifndef _STP_NOISE_TO_RIVER_LAYER_H_
#define _STP_NOISE_TO_RIVER_LAYER_H_

#include "STPCrossLayer.h"

#include <limits>

namespace {

	/**
	 * @brief STPNoiseToRiverLayer converts to land noise layer to river by sampling the same noise value
	*/
	class STPNoiseToRiverLayer : public STPCrossLayer {
	private:

		inline static STPSample_t filterRiver(STPSample_t val) noexcept {
			//giving 1/2 chance of having a river
			return val >= 2u ? 2u + (val & 1u) : val;
		}

	public:

		STPNoiseToRiverLayer(const size_t cache_size, const STPSeed_t global_seed, const STPSeed_t salt, STPLayer& parent) :
			STPCrossLayer(cache_size, global_seed, salt, parent) {

		}

		STPSample_t sample(const STPSample_t center, const STPSample_t north, const STPSample_t east,
			const STPSample_t south, const STPSample_t west, STPSeed_t) override {
			//filter the river
			//basically it's an edge detector
			const STPSample_t i = STPNoiseToRiverLayer::filterRiver(center);
			return i == STPNoiseToRiverLayer::filterRiver(north) && i == STPNoiseToRiverLayer::filterRiver(east)
				&& i == STPNoiseToRiverLayer::filterRiver(south) && i == STPNoiseToRiverLayer::filterRiver(west)
				? std::numeric_limits<STPSample_t>::max() : Reg::River.ID;
		}

	};

}
#endif//_STP_NOISE_TO_RIVER_LAYER_H_