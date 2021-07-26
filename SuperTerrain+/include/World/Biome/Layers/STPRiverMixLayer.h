#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "../STPLayer.h"
#include "../Biomes/STPBiomeRegistry.h"

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPRiverMixLayer mixes land portion with river noise, this is a merging layer
	*/
	class STPRiverMixLayer : public SuperTerrainPlus::STPDiversity::STPLayer {
	public:

		STPRiverMixLayer(Seed global_seed, Seed salt, STPLayer* parent_land, STPLayer* parent_river) : STPLayer(global_seed, salt, parent_land, parent_river) {
			//parent 0: land
			//parent 1: river noise
		}

		Sample sample(int x, int y, int z) override {
			//get the parent values
			const Sample land_val = this->getAscendant(0)->retrieve(x, y, z);
			const Sample river_val = this->getAscendant(1)->retrieve(x, y, z);

			//if the land section points to an ocean, don't touch
			//(you can't have river inside the ocean right?)
			if (STPBiomeRegistry::isOcean(land_val)) {
				return land_val;
			}

			if (river_val == STPBiomeRegistry::RIVER.getID() && STPBiomeRegistry::getPrecipitationType(land_val) == STPBiomeRegistry::STPPrecipitationType::SNOW) {
				//if this area snows, we should see the river frozen
				return STPBiomeRegistry::FROZEN_RIVER.getID();
				//vanilla minecraft has mushroom biome, but this is not realistic in real world, so we ignore
			}

			//if this area doesn't contain river
			return land_val;
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_