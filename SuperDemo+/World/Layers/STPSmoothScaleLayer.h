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

		STPSmoothScaleLayer(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer* const parent) :
			STPCrossLayer(cache_size, global_seed, salt, parent) {

		}

		Sample sample(const Sample center, const Sample north, const Sample east, const Sample south, const Sample west, const Seed local_seed) override {
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