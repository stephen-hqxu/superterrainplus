#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include <World/Biome/STPLayer.h>
#include "../Biomes/STPBiomeRegistry.h"

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPHillsLayer generates hills that are located at the central of some biomes from the noise function
	*/
	class STPHillsLayer : public SuperTerrainPlus::STPDiversity::STPLayer {
	private:

		//Chance of having a hill
		static constexpr short HILLS_CHANCE = 29;

	public:

		STPHillsLayer(Seed global_seed, Seed salt, STPLayer* land, STPLayer* noise) : STPLayer(global_seed, salt, land, noise) {
			//parent 0: land / biome
			//parent 1: noise
		}

		Sample sample(int x, int y, int z) override {
			//generate local seed 
			const Seed local_seed = this->genLocalSeed(x, z);
			//get the local rng
			const STPLayer::STPLocalRNG rng = this->getRNG(local_seed);
			//get the parent samples
			const Sample land_val = this->getAscendant(0)->retrieve(x, y, z);
			const Sample noise_val = this->getAscendant(1)->retrieve(x, y, z);
			
			//chance of having a hill
			const Sample is_hill = (noise_val - 2) % STPHillsLayer::HILLS_CHANCE;

			//1/3 chance to have a hill
			if (rng.nextVal(3) == 0 || is_hill == 0) {
				Sample l = land_val;
				//convert biomes to their respective hill biome
				if (land_val == STPBiomeRegistry::DESERT.getID()) {
					l = STPBiomeRegistry::DESERT_HILLS.getID();
				}
				else if (land_val == STPBiomeRegistry::TAIGA.getID()) {
					l = STPBiomeRegistry::TAIGA_HILLS.getID();
				}
				else if (land_val == STPBiomeRegistry::MOUNTAIN.getID()) {
					l = STPBiomeRegistry::WOODED_MOUNTAIN.getID();
				}
				else if (land_val == STPBiomeRegistry::SNOWY_TUNDRA.getID() || land_val == STPBiomeRegistry::SNOWY_TAIGA.getID()) {
					l = STPBiomeRegistry::SNOWY_MOUNTAIN.getID();
				}
				else if (land_val == STPBiomeRegistry::PLAINS.getID()) {
					l = rng.nextVal(3) == 0 ? STPBiomeRegistry::FOREST_HILLS.getID() : STPBiomeRegistry::FOREST.getID();
				}
				else if (land_val == STPBiomeRegistry::FOREST.getID()) {
					l = STPBiomeRegistry::FOREST_HILLS.getID();
				}
				else if (land_val == STPBiomeRegistry::JUNGLE.getID()) {
					l = STPBiomeRegistry::JUNGLE_HILLS.getID();
				}
				else if (land_val == STPBiomeRegistry::SAVANA.getID()) {
					l = STPBiomeRegistry::SAVANA_PLATEAU.getID();
				}
				else if (land_val == STPBiomeRegistry::SWAMP.getID()) {
					l = STPBiomeRegistry::SWAMP_HILLS.getID();
				}
				else if (land_val == STPBiomeRegistry::BADLANDS.getID()) {
					l = STPBiomeRegistry::BADLANDS_PLATEAU.getID();
				}
				//randomly generate some deep ocean as hills
				else if (land_val == STPBiomeRegistry::OCEAN.getID()) {
					l = STPBiomeRegistry::DEEP_OCEAN.getID();
				}
				else if (land_val == STPBiomeRegistry::WARM_OCEAN.getID()) {
					l = STPBiomeRegistry::DEEP_WARM_OCEAN.getID();
				}
				else if (land_val == STPBiomeRegistry::LUKEWARM_OCEAN.getID()) {
					l = STPBiomeRegistry::DEEP_LUKEWARM_OCEAN.getID();
				}
				else if (land_val == STPBiomeRegistry::COLD_OCEAN.getID()) {
					l = STPBiomeRegistry::DEEP_COLD_OCEAN.getID();
				}
				else if (land_val == STPBiomeRegistry::FROZEN_OCEAN.getID()) {
					l = STPBiomeRegistry::DEEP_FROZEN_OCEAN.getID();
				}

				//now let's add some island at the center of some ocean, given 1/3 chance of spawning
				if (STPBiomeRegistry::isOcean(land_val) && !STPBiomeRegistry::isShallowOcean(land_val) && rng.nextVal(3) == 0) {
					//filter out deep ocean
					//giving 1/2 chance of each biome, feel free to add some more...
					l = rng.nextVal(2) == 0 ? STPBiomeRegistry::PLAINS.getID() : STPBiomeRegistry::FOREST.getID();
				}

				//make sure the hill is strictly at the center of the biome, not on the edge
				if (l != land_val) {
					unsigned char m = 0x00u;
					if (land_val ==	this->getAscendant(0)->retrieve(x, y, z - 1)) {
						m++;
					}
					if (land_val == this->getAscendant(0)->retrieve(x + 1, y, z)) {
						m++;
					}
					if (land_val == this->getAscendant(0)->retrieve(x - 1, y, z)) {
						m++;
					}
					if (land_val == this->getAscendant(0)->retrieve(x, y, z + 1)) {
						m++;
					}
					if (m >= 0x03u) {
						return l;
					}
				}
			}

			return land_val;
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_