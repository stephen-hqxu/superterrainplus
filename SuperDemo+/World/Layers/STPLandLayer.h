#pragma once
#ifndef _STP_LAND_LAYER_H_
#define _STP_LAND_LAYER_H_

#include "STPXCrossLayer.h"

namespace {

	/**
	 * @brief STPLandLayer adds more lands that is not closed to other land area
	*/
	class STPLandLayer : public STPXCrossLayer {
	public:

		STPLandLayer(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer& parent) :
			STPXCrossLayer(cache_size, global_seed, salt, parent) {

		}

		Sample sample(const Sample center, const Sample ne, const Sample se, const Sample sw, const Sample nw,
			const Seed local_seed) override {
			//get the local RNG
			const STPLayer::STPLocalSampler rng = this->createLocalSampler(local_seed);

			//if the centre is not ocean or all surroundings are ocean
			if (!Reg::isShallowOcean(center) || Reg::applyAll(Reg::isShallowOcean, sw, se, ne, nw)) {
				//opposite, if the centre is ocean or all surroundings are not
				if (Reg::isShallowOcean(center) || Reg::applyAll([](Sample val) -> bool {
					return !Reg::isShallowOcean(val); 
					}, sw, se, ne, nw) || rng.nextValue(5) != 0u) {
					//then we have 1/5 chance to return the biome pointing to centre
					return center;
				}
				
				//expand the land section if it's next to other lands
				if (Reg::isShallowOcean(nw)) {
					return Reg::CAS(center, Reg::Forest.ID, nw);
				}
				if (Reg::isShallowOcean(sw)) {
					return Reg::CAS(center, Reg::Forest.ID, sw);
				}
				if (Reg::isShallowOcean(ne)) {
					return Reg::CAS(center, Reg::Forest.ID, ne);
				}
				if (Reg::isShallowOcean(se)) {
					return Reg::CAS(center, Reg::Forest.ID, se);
				}

				return center;
			}

			Sample i = 1u, j = 1u;
			//if we are surrounded by ocean, create lands with ever-decreased chance
			if (!Reg::isShallowOcean(nw) && rng.nextValue(i++) == 0) {
				j = nw;
			}
			if (!Reg::isShallowOcean(ne) && rng.nextValue(i++) == 0) {
				j = ne;
			}
			if (!Reg::isShallowOcean(sw) && rng.nextValue(i++) == 0) {
				j = sw;
			}
			if (!Reg::isShallowOcean(se) && rng.nextValue(i) == 0) {
				j = se;
			}

			if (rng.nextValue(3) == 0) {
				return j;
			}

			return j == Reg::Forest.ID ? Reg::Forest.ID : center;
		}

	};

}
#endif//_STP_LAND_LAYER_H_