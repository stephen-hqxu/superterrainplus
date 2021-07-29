#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "STPCrossLayer.h"
#include "../Biomes/STPBiomeRegistry.h"

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPIslandLayer adds more lands if the near-neighbors are all ocean, a.k.a., remove too much ocean
	*/
	class STPIslandLayer : public STPCrossLayer {
	public:

		STPIslandLayer(Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(global_seed, salt, parent){

		}

		Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed local_seed) override {
			//get the local rng
			const STPLayer::STPLocalRNG rng = this->getRNG(local_seed);

			//if we are surrounded by ocean, we have 1/2 of chance to generate a plain
			return STPBiomeRegistry::applyAll(STPBiomeRegistry::isShallowOcean, center, north, east, south, west)
				&& rng.nextVal(2) == 0 ? STPBiomeRegistry::PLAINS.getID() : center;
		}
	};
}
#endif//_STP_LAYERS_ALL_HPP_