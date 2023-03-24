#pragma once
#ifndef _STP_ISLAND_LAYER_H_
#define _STP_ISLAND_LAYER_H_

#include "STPCrossLayer.h"

namespace {

	/**
	 * @brief STPIslandLayer adds more lands if the near-neighbour are all ocean, a.k.a., remove too much ocean
	*/
	class STPIslandLayer : public STPCrossLayer {
	public:

		STPIslandLayer(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer* const parent) :
			STPCrossLayer(cache_size, global_seed, salt, parent) {

		}

		Sample sample(const Sample center, const Sample north, const Sample east, const Sample south, const Sample west,
			const Seed local_seed) override {
			//get the local RNG
			const STPLayer::STPLocalSampler rng = this->createLocalSampler(local_seed);

			//if we are surrounded by ocean, we have 1/2 of chance to generate a plain
			return Reg::applyAll(Reg::isShallowOcean, center, north, east, south, west)
				&& rng.nextValue(2) == 0 ? Reg::Plains.ID : center;
		}
	};
}
#endif//_STP_ISLAND_LAYER_H_