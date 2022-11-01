#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

//Base layer
#include <SuperTerrain+/World/Diversity/STPLayer.h>

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPXCrossLayer is an extended version of regular layer, it takes in a X cross coordinate and then sample
	*/
	class STPXCrossLayer : public SuperTerrainPlus::STPDiversity::STPLayer {
	public:

		/**
		 * @brief Init STPXCrossLayer with one parent only
		 * @param cache_size The size of cache
		 * @param global_seed World seed
		 * @param salt Random salt
		 * @param parent The previous layer
		*/
		STPXCrossLayer(size_t cache_size, Seed global_seed, Seed salt, STPLayer* parent) : STPLayer(cache_size, global_seed, salt, parent) {

		}

		Sample sample(int x, int y, int z) override {
			//sample in a X cross
			STPLayer* const asc = this->getAscendant();
			return this->sample(
				asc->retrieve(x, y, z),
				asc->retrieve(x + 1, y, z - 1),
				asc->retrieve(x + 1, y, z + 1),
				asc->retrieve(x - 1, y, z + 1),
				asc->retrieve(x - 1, y, z - 1),
				this->seedLocal(x, z)
			);
		}

		/**
		 * @brief Sample the layer in a X cross manner, return the sample point
		 * @param centre The centre coordinate of the X cross
		 * @param ne The north east coordinate of the X cross
		 * @param se The south east coordinate of the X cross
		 * @param sw The south west coordinate of the X cross
		 * @param nw The north west coordinate of the X cross
		 * @param local_seed The seed for this local generation
		 * @return Sample id or value, depended on the actual implementation.
		*/
		virtual Sample sample(Sample, Sample, Sample, Sample, Sample, Seed) = 0;

	};
}
#endif//_STP_LAYERS_ALL_HPP_