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

		STPIslandLayer(const size_t cache_size, const STPSeed_t global_seed, const STPSeed_t salt, STPLayer& parent) :
			STPCrossLayer(cache_size, global_seed, salt, parent) {

		}

		STPSample_t sample(const STPSample_t center, const STPSample_t north, const STPSample_t east,
			const STPSample_t south, const STPSample_t west, const STPSeed_t local_seed) override {
			//get the local RNG
			const STPLayer::STPLocalSampler rng = this->createLocalSampler(local_seed);

			//if we are surrounded by ocean, we have 1/2 of chance to generate a plain
			return Reg::applyAll(Reg::isShallowOcean, center, north, east, south, west)
				&& rng.nextValue(2) == 0 ? Reg::Plains.ID : center;
		}
	};
}
#endif//_STP_ISLAND_LAYER_H_