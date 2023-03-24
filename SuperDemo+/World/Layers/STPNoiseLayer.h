#pragma once
#ifndef _STP_NOISE_LAYER_H_
#define _STP_NOISE_LAYER_H_

namespace {

	/**
	 * @brief STPNoiseLayer generates random value on non-ocean biomes, this will be mainly used for river network generation
	*/
	class STPNoiseLayer : public SuperTerrainPlus::STPDiversity::STPLayer {
	public:

		STPNoiseLayer(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer& parent) :
			STPLayer(cache_size, global_seed, salt, { parent }) {
			//noise layer will overwrite previous interpretation, this is a new chain of layers
		}

		Sample sample(const int x, const int y, const int z) override {
			//get the local generator
			const STPLayer::STPLocalSampler rng = this->createLocalSampler(x, z);

			//value from the previous layer
			const Sample val = this->getAscendant().retrieve(x, y, z);
			//leaving ocean untouched, given a random noise value for the river generation layer
			return Reg::isShallowOcean(val) ? val : rng.nextValue(29999) + 2;
		}

	};

}
#endif//_STP_NOISE_LAYER_H_