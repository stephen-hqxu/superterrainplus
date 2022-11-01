#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "STPCrossLayer.h"

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPSmoothScaleLayer scales previous sample smoothly, and only picks sample that are equals
	*/
	class STPSmoothScaleLayer : public STPCrossLayer {
	public:

		STPSmoothScaleLayer(size_t cache_size, Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(cache_size, global_seed, salt, parent) {

		}

		Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed local_seed) {
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
#endif//_STP_LAYERS_ALL_HPP_
