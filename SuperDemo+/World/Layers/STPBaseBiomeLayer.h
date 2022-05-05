#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include <SuperTerrain+/World/Diversity/STPLayer.h>
#include "../Biomes/STPBiomeRegistry.h"

#include <array>

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPBaseBiomeLayer starts to add biomes based on the climate, and interpret temperature and precipitation to the actual biome
	*/
	class STPBaseBiomeLayer : public SuperTerrainPlus::STPDiversity::STPLayer {
	private:

		template<size_t S>
		using STPBiomeList = std::array<Sample, S>;

		//A table of interpretation
		//If we want higher chance of spawning for specific biomes, just repeat that one
		//Since biome ids are loaded after runtime, so we can't make it static

		const STPBiomeList<4u> DryBiomes = {
			STPBiomeRegistry::Desert.ID,
			STPBiomeRegistry::Savannah.ID,
			STPBiomeRegistry::Plains.ID,
			STPBiomeRegistry::Badlands.ID
		};

		const STPBiomeList<5u> TemperateBiomes = {
			STPBiomeRegistry::Jungle.ID,
			STPBiomeRegistry::Forest.ID,
			STPBiomeRegistry::Mountain.ID,
			STPBiomeRegistry::Plains.ID,
			STPBiomeRegistry::Swamp.ID
		};

		const STPBiomeList<4u> CoolBiomes = {
			STPBiomeRegistry::Plains.ID,
			STPBiomeRegistry::Mountain.ID,
			STPBiomeRegistry::Forest.ID,
			STPBiomeRegistry::Taiga.ID
		};

		const STPBiomeList<3u> SnowyBiomes = {
			STPBiomeRegistry::SnowyTaiga.ID,
			STPBiomeRegistry::SnowyTundra.ID,
			STPBiomeRegistry::SnowyMountain.ID
		};

	public:

		STPBaseBiomeLayer(Seed global_seed, Seed salt, STPLayer* parent) : STPLayer(global_seed, salt, parent) {
			//parent:: climate layer
		}

		Sample sample(int x, int y, int z) override {
			//set the local seed
			const Seed local_seed = this->genLocalSeed(x, z);
			//get the local RNG
			STPLayer::STPLocalRNG rng = this->getRNG(local_seed);
			//get the climate for this local coordinate
			const Sample climate = this->getAscendant()->retrieve(x, y, z);

			//if it's ocean, we should leave it untouched
			if (STPBiomeRegistry::isOcean(climate)) {
				return climate;
			}
			
			//interpretation, compared to vanilla Minecraft, special climate has been removed, every biomes have the equal chance of spawning
			if (climate == STPBiomeRegistry::Plains.ID) {
				//dry and hot biome
				return this->DryBiomes[rng.nextVal(static_cast<Sample>(this->DryBiomes.size()))];
			}
			if (climate == STPBiomeRegistry::Desert.ID) {
				//temperate biome
				return this->TemperateBiomes[rng.nextVal(static_cast<Sample>(this->TemperateBiomes.size()))];
			}
			if (climate == STPBiomeRegistry::Mountain.ID) {
				//cool biome
				return this->CoolBiomes[rng.nextVal(static_cast<Sample>(this->CoolBiomes.size()))];
			}
			if (climate == STPBiomeRegistry::Forest.ID) {
				//snowy and cold biome
				return this->SnowyBiomes[rng.nextVal(static_cast<Sample>(this->SnowyBiomes.size()))];
			}

			//this usually won't happen, but just in case
			return climate;
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_