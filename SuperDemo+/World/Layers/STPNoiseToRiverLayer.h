#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "STPCrossLayer.h"

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPNoiseToRiverLayer converts to land noise layer to river by sampling the same noise value
	*/
	class STPNoiseToRiverLayer : public STPCrossLayer {
	private:

		inline static Sample filterRiver(Sample val) {
			//giving 1/2 chance of having a river
			return val >= 2 ? 2 + (val & 1) : val;
		}

	public:

		STPNoiseToRiverLayer(Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(global_seed, salt, parent) {

		}

		Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed local_seed) override {
			//filter the river
			//bascially it's an edge detector
			const Sample i = STPNoiseToRiverLayer::filterRiver(center);
			return i == STPNoiseToRiverLayer::filterRiver(north) && i == STPNoiseToRiverLayer::filterRiver(east)
				&& i == STPNoiseToRiverLayer::filterRiver(south) && i == STPNoiseToRiverLayer::filterRiver(west)
				? 0xFFFFu : STPBiomeRegistry::RIVER.getID();
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_