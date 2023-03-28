#pragma once
#ifndef _STP_EASE_EDGE_LAYER_H_
#define _STP_EASE_EDGE_LAYER_H_

#include "STPCrossLayer.h"

#include <type_traits>

namespace {

	/**
	 * @brief STPEaseEdgeLayer inserts temperate biomes between layers if the transition is not suitable
	*/
	class STPEaseEdgeLayer : public STPCrossLayer {
	private:

		template<typename... S>
		inline static bool anyMatch(const STPDemo::STPBiome& biome, const S... id) noexcept {
			static_assert(std::conjunction<std::is_same<S, STPSample_t>...>::value, "Only biome IDs can be matched");
			return ((biome.ID == id) || ...);
		}

	public:

		STPEaseEdgeLayer(const size_t cache_size, const STPSeed_t global_seed, const STPSeed_t salt, STPLayer& parent) :
			STPCrossLayer(cache_size, global_seed, salt, parent) {

		}

		STPSample_t sample(const STPSample_t center, const STPSample_t north, const STPSample_t east,
			const STPSample_t south, const STPSample_t west, STPSeed_t) override {
			//replace the edge by something else if it has conflict
			if (center == Reg::Desert.ID && STPEaseEdgeLayer::anyMatch(Reg::SnowyTundra, north, east, south, west)) {
				return Reg::WoodedMountain.ID;
			}
			if (center == Reg::Swamp.ID
				&& (STPEaseEdgeLayer::anyMatch(Reg::Desert, north, east, south, west)
				|| STPEaseEdgeLayer::anyMatch(Reg::SnowyTundra, north, east, south, west)
				|| STPEaseEdgeLayer::anyMatch(Reg::SnowyTaiga, north, east, south, west))) {
				return Reg::Plains.ID;
			}

			//otherwise do nothing
			return center;
		}

	};

}
#endif//_STP_EASE_EDGE_LAYER_H_