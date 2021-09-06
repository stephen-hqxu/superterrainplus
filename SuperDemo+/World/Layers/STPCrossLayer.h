#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

//Base layer
#include <SuperTerrain+/World/Diversity/STPLayer.h>

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPCrossLayer is an extended version of regular layer, it takes in a cross coordinate and then sample
	*/
	class STPCrossLayer : public SuperTerrainPlus::STPDiversity::STPLayer {
	public:

		/**
		 * @brief Init STPCrossLayer with one parent only
		 * @param global_seed World seed
		 * @param salt Random salt
		 * @param parent The previous layer
		*/
		STPCrossLayer(Seed global_seed, Seed salt, STPLayer* parent) : STPLayer(global_seed, salt, parent) {

		}

		Sample sample(int x, int y, int z) override {
			//sample in a cross
			STPLayer* const asc = this->getAscendant();
			return this->sample(
				asc->retrieve(x, y, z),
				asc->retrieve(x, y, z - 1),
				asc->retrieve(x + 1, y, z),
				asc->retrieve(x, y, z + 1),
				asc->retrieve(x - 1, y, z),
				this->genLocalSeed(x, z)
			);
		}

		/**
		 * @brief Sample the layer in a cross manner, return the sample point
		 * @param center The center coodinate of the cross
		 * @param north The northern pixel of the cross
		 * @param east The east pixel of the cross
		 * @param south The south pixel of the cross
		 * @param west The west pixel of the cross
		 * @param local_seed The seed for this local generation
		 * @return Sample id or value, depended on the actual implementation.
		*/
		virtual Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed local_seed) = 0;

	};
}
#endif//_STP_LAYERS_ALL_HPP_