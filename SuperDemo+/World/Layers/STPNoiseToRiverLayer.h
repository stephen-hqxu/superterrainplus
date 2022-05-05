#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "STPCrossLayer.h"

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

		Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed) override {
			//filter the river
			//basically it's an edge detector
			const Sample i = STPNoiseToRiverLayer::filterRiver(center);
			return i == STPNoiseToRiverLayer::filterRiver(north) && i == STPNoiseToRiverLayer::filterRiver(east)
				&& i == STPNoiseToRiverLayer::filterRiver(south) && i == STPNoiseToRiverLayer::filterRiver(west)
				? 0xFFFFu : STPBiomeRegistry::River.ID;
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_