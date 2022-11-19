#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "STPCrossLayer.h"
#include "../Biomes/STPBiomeRegistry.h"

#include <type_traits>

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPEaseEdgeLayer inserts temperate biomes between layers if the transition is not suitable
	*/
	class STPEaseEdgeLayer : public STPCrossLayer {
	private:

		template<typename... S>
		inline static bool anyMatch(const STPBiome& biome, const S... id) noexcept {
			static_assert(std::conjunction<std::is_same<S, Sample>...>::value, "Only biome IDs can be matched");

			return ((biome.ID == id) || ...);
		}

	public:

		STPEaseEdgeLayer(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer* const parent) :
			STPCrossLayer(cache_size, global_seed, salt, parent) {

		}

		Sample sample(const Sample center, const Sample north, const Sample east, const Sample south, const Sample west, Seed) override {
			//replace the edge by something else if it has conflict
			if (center == STPBiomeRegistry::Desert.ID && STPEaseEdgeLayer::anyMatch(STPBiomeRegistry::SnowyTundra, north, east, south, west)) {
				return STPBiomeRegistry::WoodedMountain.ID;
			}
			if (center == STPBiomeRegistry::Swamp.ID
				&& (STPEaseEdgeLayer::anyMatch(STPBiomeRegistry::Desert, north, east, south, west)
				|| STPEaseEdgeLayer::anyMatch(STPBiomeRegistry::SnowyTundra, north, east, south, west)
				|| STPEaseEdgeLayer::anyMatch(STPBiomeRegistry::SnowyTaiga, north, east, south, west))) {
				return STPBiomeRegistry::Plains.ID;
			}

			//otherwise do nothing
			return center;
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_