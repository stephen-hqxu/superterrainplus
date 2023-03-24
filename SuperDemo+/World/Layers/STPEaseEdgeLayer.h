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
			static_assert(std::conjunction<std::is_same<S, Sample>...>::value, "Only biome IDs can be matched");

			return ((biome.ID == id) || ...);
		}

	public:

		STPEaseEdgeLayer(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer* const parent) :
			STPCrossLayer(cache_size, global_seed, salt, parent) {

		}

		Sample sample(const Sample center, const Sample north, const Sample east, const Sample south, const Sample west, Seed) override {
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