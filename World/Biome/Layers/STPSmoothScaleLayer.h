#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "STPCrossLayer.h"

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPBiome::Seed;
	using SuperTerrainPlus::STPBiome::Sample;

	/**
	 * @brief STPSmoothScaleLayer scales previous sample smoothly, and only picks sample that are equals
	*/
	class STPSmoothScaleLayer : public STPCrossLayer {
	public:

		STPSmoothScaleLayer(Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(global_seed, salt, parent) {

		}

		Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed local_seed) {
			//set local rng
			const STPLayer::STPLocalRNG rng = this->getRNG(local_seed);

			const bool xMatch = west == east;
			const bool zMatch = north == south;
			if (xMatch == zMatch) {
				//if both of them are the same, pick randomly
				//if not, do not touch the center value
				return xMatch ? rng.choose(west, north) : center;
			}
			//otherwise we pick the side that is the same
			return xMatch ? west : north;
		}

	};
}
#endif//_STP_LAYERS_ALL_HPP_
