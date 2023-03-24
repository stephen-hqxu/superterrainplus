#pragma once
#ifndef _STP_RIVER_MIX_LAYER_H_
#define _STP_RIVER_MIX_LAYER_H_

namespace {

	/**
	 * @brief STPRiverMixLayer mixes land portion with river noise, this is a merging layer
	*/
	class STPRiverMixLayer : public STPLayer {
	public:

		STPRiverMixLayer(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer& parent_land, STPLayer& parent_river) :
			STPLayer(cache_size, global_seed, salt, { parent_land, parent_river }) {
			//parent 0: land
			//parent 1: river noise
		}

		Sample sample(const int x, const int y, const int z) override {
			//get the parent values
			const Sample land_val = this->getAscendant(0).retrieve(x, y, z);
			const Sample river_val = this->getAscendant(1).retrieve(x, y, z);

			//if the land section points to an ocean, don't touch
			//(you can't have river inside the ocean right?)
			if (Reg::isOcean(land_val)) {
				return land_val;
			}

			if (river_val == Reg::River.ID && Reg::getPrecipitationType(land_val) == Reg::STPPrecipitationType::SNOW) {
				//if this area snows, we should see the river frozen
				return Reg::FrozenRiver.ID;
				//vanilla Minecraft has mushroom biome, but this is not realistic in real world, so we ignore
			}

			//if this area doesn't contain river
			return land_val;
		}

	};

}
#endif//_STP_RIVER_MIX_LAYER_H_