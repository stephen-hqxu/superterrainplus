#pragma once
#ifndef _STP_CROSS_LAYER_H_
#define _STP_CROSS_LAYER_H_

namespace {

	/**
	 * @brief STPCrossLayer is an extended version of regular layer, it takes in a cross coordinate and then sample
	*/
	class STPCrossLayer : public STPLayer {
	public:

		/**
		 * @brief Init STPCrossLayer with one parent only
		 * @param cache_size Cache size
		 * @param global_seed World seed
		 * @param salt Random salt
		 * @param parent The previous layer
		*/
		STPCrossLayer(const size_t cache_size, const STPSeed_t global_seed, const STPSeed_t salt, STPLayer& parent) :
			STPLayer(cache_size, global_seed, salt, { parent }) {

		}

		STPSample_t sample(const int x, const int y, const int z) override {
			//sample in a cross
			STPLayer& asc = this->getAscendant();
			return this->sample(
				asc.retrieve(x, y, z),
				asc.retrieve(x, y, z - 1),
				asc.retrieve(x + 1, y, z),
				asc.retrieve(x, y, z + 1),
				asc.retrieve(x - 1, y, z),
				this->seedLocal(x, z)
			);
		}

		/**
		 * @brief Sample the layer in a cross manner, return the sample point
		 * @param centre The centre coordinate of the cross
		 * @param north The northern pixel of the cross
		 * @param east The east pixel of the cross
		 * @param south The south pixel of the cross
		 * @param west The west pixel of the cross
		 * @param local_seed The seed for this local generation
		 * @return Sample id or value, depended on the actual implementation.
		*/
		virtual STPSample_t sample(STPSample_t, STPSample_t, STPSample_t, STPSample_t, STPSample_t, STPSeed_t) = 0;

	};
}
#endif//_STP_CROSS_LAYER_H_