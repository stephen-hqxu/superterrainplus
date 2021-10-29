#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "STPXCrossLayer.h"
#include "../Biomes/STPBiomeRegistry.h"

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPLandLayer adds more lands that is not closed to other land area
	*/
	class STPLandLayer : public STPXCrossLayer {
	public:

		STPLandLayer(Seed global_seed, Seed salt, STPLayer* parent) : STPXCrossLayer(global_seed, salt, parent) {

		}

		Sample sample(Sample center, Sample ne, Sample se, Sample sw, Sample nw, Seed local_seed) override {
			//get the local rng
			const STPLayer::STPLocalRNG rng = this->getRNG(local_seed);

			//if the center is not ocean or all surroundings are ocean
			if (!STPBiomeRegistry::isShallowOcean(center) || STPBiomeRegistry::applyAll(STPBiomeRegistry::isShallowOcean, sw, se, ne, nw)) {
				//opposite, if the center is ocean or all surroundings are not
				if (STPBiomeRegistry::isShallowOcean(center) || STPBiomeRegistry::applyAll([](Sample val) -> bool {
					return !STPBiomeRegistry::isShallowOcean(val); 
					}, sw, se, ne, nw) || rng.nextVal(5) != 0u) {
					//then we have 1/5 chance to return the biome pointing to center
					return center;
				}
				
				//expand the land section if it's next to other lands
				if (STPBiomeRegistry::isShallowOcean(nw)) {
					return STPBiomeRegistry::CAS(center, STPBiomeRegistry::FOREST.getID(), nw);
				}
				if (STPBiomeRegistry::isShallowOcean(sw)) {
					return STPBiomeRegistry::CAS(center, STPBiomeRegistry::FOREST.getID(), sw);
				}
				if (STPBiomeRegistry::isShallowOcean(ne)) {
					return STPBiomeRegistry::CAS(center, STPBiomeRegistry::FOREST.getID(), ne);
				}
				if (STPBiomeRegistry::isShallowOcean(se)) {
					return STPBiomeRegistry::CAS(center, STPBiomeRegistry::FOREST.getID(), se);
				}

				return center;
			}

			Sample i = 1u, j = 1u;
			//if we are surrounded by ocean, create lands with ever-decreased chance
			if (!STPBiomeRegistry::isShallowOcean(nw) && rng.nextVal(i++) == 0) {
				j = nw;
			}
			if (!STPBiomeRegistry::isShallowOcean(ne) && rng.nextVal(i++) == 0) {
				j = ne;
			}
			if (!STPBiomeRegistry::isShallowOcean(sw) && rng.nextVal(i++) == 0) {
				j = sw;
			}
			if (!STPBiomeRegistry::isShallowOcean(se) && rng.nextVal(i) == 0) {
				j = se;
			}

			if (rng.nextVal(3) == 0) {
				return j;
			}

			return j == STPBiomeRegistry::FOREST.getID() ? STPBiomeRegistry::FOREST.getID() : center;
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_