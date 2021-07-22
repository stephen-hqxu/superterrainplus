#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "STPCrossLayer.h"
#include "../Biomes/STPBiomeRegistry.h"

#include <type_traits>

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPEaseEdgeLayer inserts temperate biomes between layers if the transition is not suitable
	*/
	class STPEaseEdgeLayer : public STPCrossLayer {
	private:

		template<typename... S>
		inline static bool anyMatch(const STPBiome& biome, S... id) {
			static_assert(std::conjunction<std::is_same<S, Sample>...>::value, "Only biome IDs can be matched");

			return ((biome.getID() == id) || ...);
		}

	public:

		STPEaseEdgeLayer(Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(global_seed, salt, parent) {

		}

		Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed local_seed) override {
			//replace the edge by something else if it has conflict
			if (center == STPBiomeRegistry::DESERT.getID() && STPEaseEdgeLayer::anyMatch(STPBiomeRegistry::SNOWY_TUNDRA, north, east, south, west)) {
				return STPBiomeRegistry::WOODED_MOUNTAIN.getID();
			}
			if (center == STPBiomeRegistry::SWAMP.getID()
				&& (STPEaseEdgeLayer::anyMatch(STPBiomeRegistry::DESERT, north, east, south, west)
				|| STPEaseEdgeLayer::anyMatch(STPBiomeRegistry::SNOWY_TUNDRA, north, east, south, west)
				|| STPEaseEdgeLayer::anyMatch(STPBiomeRegistry::SNOWY_TAIGA, north, east, south, west))) {
				return STPBiomeRegistry::PLAINS.getID();
			}

			//otherwise do nothing
			return center;
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_