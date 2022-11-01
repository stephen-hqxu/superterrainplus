#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include <SuperTerrain+/World/Diversity/STPLayer.h>

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPScaleLayer scales the current layer and randomly choose the neighbouring cell to fill the new cells
	*/
	class STPScaleLayer : public SuperTerrainPlus::STPDiversity::STPLayer {
	public:

		/**
		 * @brief STPScaleType defines the type of scaling
		*/
		enum class STPScaleType : unsigned char {
			//Normal scaling
			NORMAL = 0x00,
			//Fuzzy scaling
			FUZZY = 0xFF
		};

	private:

		//The scaling type for this layer
		const STPScaleType Type;

		Sample sample(Sample center, Sample e, Sample s, Sample se, const STPLayer::STPLocalRNG& rng) {
			//choose randomly between each cell
			const Sample ret = rng.choose(center, e, s, se);

			if (this->Type == STPScaleType::FUZZY) {
				return ret;
			}

			//if the opposite cells are the same, return the same sample
			if (e == s && e == se) {
				return e;
			}
			//if the neighbour cells are the same, return the same sample
			if (center == e && (center == se || s != se)) {
				return center;
			}
			if (center == s && (center == se || e != se)) {
				return center;
			}
			if (center == se && e != s) {
				return center;
			}
			if (e == s && center != se) {
				return e;
			}
			if (e == se && center != s) {
				return e;
			}
			if (s == se && center != e) {
				return s;
			}

			//if none of them are the same, randomly choose between each of them
			return ret;
		}

	public:

		STPScaleLayer(size_t cache_size, Seed global_seed, Seed salt, STPScaleType type, STPLayer* parent) :
			STPLayer(cache_size, global_seed, salt, parent), Type(type) {
			//parent: undefined
		}

		Sample sample(int x, int y, int z) override {
			//get the sample of the neighbour cell
			const Sample i = this->getAscendant()->retrieve(x >> 1, y, z >> 1);
			const int xb = x & 1, zb = z & 1;
			//reset local seed
			const Seed local_seed = this->genLocalSeed(x & -2, z & -2);
			//get local RNG
			const STPLayer::STPLocalRNG rng = this->getRNG(local_seed);

			if (xb == 0 && zb == 0) {
				//if the zoomed cell is exactly the current cell, return current value
				return i;
			}

			//otherwise, we need to randomly choose between neighbouring values
			const Sample l = this->getAscendant()->retrieve(x >> 1, y, (z + 1) >> 1);
			const Sample m = rng.choose(i, l);

			if (xb == 0) {
				return m;
			}

			const Sample n = this->getAscendant()->retrieve((x + 1) >> 1, y, z >> 1);
			const Sample o = rng.choose(i, n);

			if (zb == 0) {
				return o;
			}

			const Sample p = this->getAscendant()->retrieve((x + 1) >> 1, y, (z + 1) >> 1);
			return this->sample(i, n, l, p, rng);

		}

	};
}
#endif//_STP_LAYERS_ALL_HPP_