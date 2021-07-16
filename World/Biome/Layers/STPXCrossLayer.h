#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

//Base layer
#include "../STPLayer.h"

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
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
		 * @param global_seed World seed
		 * @param salt Random salt
		 * @param parent The previous layer
		*/
		STPXCrossLayer(Seed global_seed, Seed salt, STPLayer* parent) : STPLayer(global_seed, salt, parent) {

		}

		Sample sample(int x, int y, int z) override {
			//sample in a X cross
			STPLayer* const asc = this->getAscendant();
			return this->sample(
				asc->sample_cached(x, y, z),
				asc->sample_cached(x + 1, y, z - 1),
				asc->sample_cached(x + 1, y, z + 1),
				asc->sample_cached(x - 1, y, z + 1),
				asc->sample_cached(x - 1, y, z - 1),
				this->genLocalSeed(x, z)
			);
		}

		/**
		 * @brief Sample the layer in a X cross manner, return the sample point
		 * @param center The center coordinate of the X cross
		 * @param ne The north east coordinate of the X cross
		 * @param se The south east coordinate of the X cross
		 * @param sw The south west coordinate of the X cross
		 * @param nw The north west coordinate of the X cross
		 * @param local_seed The seed for this local generation
		 * @return Sample id or value, depended on the actual implementation.
		*/
		virtual Sample sample(Sample center, Sample ne, Sample se, Sample sw, Sample nw, Seed local_seed) = 0;

	};
}
#endif//_STP_LAYERS_ALL_HPP_