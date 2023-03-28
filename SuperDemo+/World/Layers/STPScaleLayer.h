#pragma once
#ifndef _STP_SCALE_LAYER_H_
#define _STP_SCALE_LAYER_H_

namespace {

	/**
	 * @brief STPScaleLayer scales the current layer and randomly choose the neighbouring cell to fill the new cells
	*/
	class STPScaleLayer : public STPLayer {
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

		STPSample_t sample(const STPSample_t center, const STPSample_t e, const STPSample_t s, const STPSample_t se,
			const STPLayer::STPLocalSampler& rng) noexcept {
			//choose randomly between each cell
			const STPSample_t ret = rng.choose(center, e, s, se);

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

		STPScaleLayer(const size_t cache_size, const STPSeed_t global_seed, const STPSeed_t salt, const STPScaleType type, STPLayer& parent) :
			STPLayer(cache_size, global_seed, salt, { parent }),
			Type(type) {
			//parent: undefined
		}

		STPSample_t sample(const int x, const int y, const int z) override {
			//get the sample of the neighbour cell
			const STPSample_t i = this->getAscendant().retrieve(x >> 1, y, z >> 1);
			const int xb = x & 1, zb = z & 1;
			//get local RNG
			const STPLayer::STPLocalSampler rng = this->createLocalSampler(x & -2, z & -2);

			if (xb == 0 && zb == 0) {
				//if the zoomed cell is exactly the current cell, return current value
				return i;
			}

			//otherwise, we need to randomly choose between neighbouring values
			const STPSample_t l = this->getAscendant().retrieve(x >> 1, y, (z + 1) >> 1);
			const STPSample_t m = rng.choose(i, l);

			if (xb == 0) {
				return m;
			}

			const STPSample_t n = this->getAscendant().retrieve((x + 1) >> 1, y, z >> 1);
			const STPSample_t o = rng.choose(i, n);

			if (zb == 0) {
				return o;
			}

			const STPSample_t p = this->getAscendant().retrieve((x + 1) >> 1, y, (z + 1) >> 1);
			return this->sample(i, n, l, p, rng);

		}

	};
}
#endif//_STP_SCALE_LAYER_H_