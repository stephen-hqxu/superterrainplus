#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "../STPLayer.h"
#include "../Biomes/STPBiomeRegistry.h"

#include <vector>

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPBaseBiomeLayer starts to add biomes based on the climate, and interprete temperature and precipitation to the actual biome
	*/
	class STPBaseBiomeLayer : public SuperTerrainPlus::STPDiversity::STPLayer {
	private:

		typedef const std::vector<Sample> BiomeList;

		//A table of interpretation
		//If we want higher chance of spawning for specific biomes, just repeat that one
		//Since biome ids are loaded after runtime, so we can't make it static

		BiomeList DRY_BIOMES = {
			STPBiomeRegistry::DESERT.getID(),
			STPBiomeRegistry::SAVANA.getID(),
			STPBiomeRegistry::PLAINS.getID(),
			STPBiomeRegistry::BADLANDS.getID()
		};

		BiomeList TEMPERATE_BIOMES = {
			STPBiomeRegistry::JUNGLE.getID(),
			STPBiomeRegistry::FOREST.getID(),
			STPBiomeRegistry::MOUNTAIN.getID(),
			STPBiomeRegistry::PLAINS.getID(),
			STPBiomeRegistry::SWAMP.getID()
		};

		BiomeList COOL_BIOMES = {
			STPBiomeRegistry::PLAINS.getID(),
			STPBiomeRegistry::MOUNTAIN.getID(),
			STPBiomeRegistry::FOREST.getID(),
			STPBiomeRegistry::TAIGA.getID()
		};

		BiomeList SNOWY_BIOMES = {
			STPBiomeRegistry::SNOWY_TAIGA.getID(),
			STPBiomeRegistry::SNOWY_TUNDRA.getID(),
			STPBiomeRegistry::SNOWY_MOUNTAIN.getID()
		};

	public:

		STPBaseBiomeLayer(Seed global_seed, Seed salt, STPLayer* parent) : STPLayer(global_seed, salt, parent) {
			//parent:: climate layer
		}

		Sample sample(int x, int y, int z) override {
			//set the local seed
			const Seed local_seed = this->genLocalSeed(x, z);
			//get the local rng
			STPLayer::STPLocalRNG rng = this->getRNG(local_seed);
			//get the climate for this local coordinate
			const Sample climate = this->getAscendant()->retrieve(x, y, z);

			//if it's ocean, we should leave it untouched
			if (STPBiomeRegistry::isOcean(climate)) {
				return climate;
			}
			
			//interpretation, compared to vanilla minecraft, special climate has been removed, every biomes have the equal chance of spawning
			if (climate == STPBiomeRegistry::PLAINS.getID()) {
				//dry and hot biome
				return this->DRY_BIOMES[rng.nextVal(static_cast<Sample>(this->DRY_BIOMES.size()))];
			}
			if (climate == STPBiomeRegistry::DESERT.getID()) {
				//temperate biome
				return this->TEMPERATE_BIOMES[rng.nextVal(static_cast<Sample>(this->TEMPERATE_BIOMES.size()))];
			}
			if (climate == STPBiomeRegistry::MOUNTAIN.getID()) {
				//cool biome
				return this->COOL_BIOMES[rng.nextVal(static_cast<Sample>(this->COOL_BIOMES.size()))];
			}
			if (climate == STPBiomeRegistry::FOREST.getID()) {
				//snowy and cold biome
				return this->SNOWY_BIOMES[rng.nextVal(static_cast<Sample>(this->SNOWY_BIOMES.size()))];
			}

			//this usually won't happen, but just in case
			return climate;
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_