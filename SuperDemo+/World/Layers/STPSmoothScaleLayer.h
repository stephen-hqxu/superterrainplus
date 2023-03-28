#pragma once
#ifndef _STP_SMOOTH_SCALE_LAYER_H_
#define _STP_SMOOTH_SCALE_LAYER_H_

#include "STPCrossLayer.h"

namespace {

	/**
	 * @brief STPSmoothScaleLayer scales previous sample smoothly, and only picks sample that are equals
	*/
	class STPSmoothScaleLayer : public STPCrossLayer {
	public:

		STPSmoothScaleLayer(const size_t cache_size, const STPSeed_t global_seed, const STPSeed_t salt, STPLayer& parent) :
			STPCrossLayer(cache_size, global_seed, salt, parent) {

		}

		STPSample_t sample(const STPSample_t center, const STPSample_t north, const STPSample_t east,
			const STPSample_t south, const STPSample_t west, const STPSeed_t local_seed) override {
			//set local RNG
			const STPLayer::STPLocalSampler rng = this->createLocalSampler(local_seed);

			const bool xMatch = west == east;
			const bool zMatch = north == south;
			if (xMatch == zMatch) {
				//if both of them are the same, pick randomly
				//if not, do not touch the centre value
				return xMatch ? rng.choose(west, north) : center;
			}
			//otherwise we pick the side that is the same
			return xMatch ? west : north;
		}

	};
}
#endif//_STP_SMOOTH_SCALE_LAYER_H_