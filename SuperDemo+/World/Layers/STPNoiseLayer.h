#pragma once
#ifndef _STP_NOISE_LAYER_H_
#define _STP_NOISE_LAYER_H_

#include <limits>

namespace {

	/**
	 * @brief STPNoiseLayer generates random value on non-ocean biomes, this will be mainly used for river network generation
	*/
	class STPNoiseLayer : public SuperTerrainPlus::STPDiversity::STPLayer {
	public:

		STPNoiseLayer(const size_t cache_size, const STPSeed_t global_seed, const STPSeed_t salt, STPLayer& parent) :
			STPLayer(cache_size, global_seed, salt, { parent }) {
			//noise layer will overwrite previous interpretation, this is a new chain of layers
		}

		STPSample_t sample(const int x, const int y, const int z) override {
			//get the local generator
			const STPLayer::STPLocalSampler rng = this->createLocalSampler(x, z);

			//value from the previous layer
			const STPSample_t val = this->getAscendant().retrieve(x, y, z);
			//leaving ocean untouched, given a random noise value for the river generation layer
			return Reg::isShallowOcean(val) ? val : rng.nextValue(std::numeric_limits<STPSample_t>::max() / 2u) + 2u;
		}

	};

}
#endif//_STP_NOISE_LAYER_H_