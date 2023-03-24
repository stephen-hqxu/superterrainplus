#pragma once
#ifndef _STP_CONTINENT_LAYER_H_
#define _STP_CONTINENT_LAYER_H_

namespace {

	/**
	 * @brief STPContinentLayer is the first layer of the biome generation chain, it generates land and ocean section
	*/
	class STPContinentLayer : public STPLayer {
	public:

		STPContinentLayer(const size_t cache_size, const Seed global_seed, const Seed salt) : STPLayer(cache_size, global_seed, salt) {

		}

		Sample sample(const int x, int, const int z) override {
			//get the RNG for this coordinate
			const STPLayer::STPLocalSampler rng = this->createLocalSampler(x, z);

			//we give 1/10 chance for land
			return rng.nextValue(10u) == 0 ? Reg::Plains.ID : Reg::Ocean.ID;
		}
	};
}
#endif//_STP_CONTINENT_LAYER_H_