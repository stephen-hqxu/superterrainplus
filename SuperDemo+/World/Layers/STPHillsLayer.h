#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include <SuperTerrain+/World/Diversity/STPLayer.h>
#include "../Biomes/STPBiomeRegistry.h"

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

		STPHillsLayer(size_t cache_size, Seed global_seed, Seed salt, STPLayer* land, STPLayer* noise) : STPLayer(cache_size, global_seed, salt, land, noise) {
			//parent 0: land / biome
			//parent 1: noise
		}

		Sample sample(int x, int y, int z) override {
			//get the local RNG
			const STPLayer::STPLocalSampler rng = this->createLocalSampler(x, z);
			//get the parent samples
			const Sample land_val = this->getAscendant(0)->retrieve(x, y, z);
			const Sample noise_val = this->getAscendant(1)->retrieve(x, y, z);
			
			//chance of having a hill
			const Sample is_hill = (noise_val - 2) % STPHillsLayer::HILLS_CHANCE;

			//1/3 chance to have a hill
			if (rng.nextValue(3) == 0 || is_hill == 0) {
				Sample l = land_val;
				//convert biomes to their respective hill biome
				if (land_val == STPBiomeRegistry::Desert.ID) {
					l = STPBiomeRegistry::DesertHills.ID;
				}
				else if (land_val == STPBiomeRegistry::Taiga.ID) {
					l = STPBiomeRegistry::TaigaHills.ID;
				}
				else if (land_val == STPBiomeRegistry::Mountain.ID) {
					l = STPBiomeRegistry::WoodedMountain.ID;
				}
				else if (land_val == STPBiomeRegistry::SnowyTundra.ID || land_val == STPBiomeRegistry::SnowyTaiga.ID) {
					l = STPBiomeRegistry::SnowyMountain.ID;
				}
				else if (land_val == STPBiomeRegistry::Plains.ID) {
					l = rng.nextValue(3) == 0 ? STPBiomeRegistry::ForestHills.ID : STPBiomeRegistry::Forest.ID;
				}
				else if (land_val == STPBiomeRegistry::Forest.ID) {
					l = STPBiomeRegistry::ForestHills.ID;
				}
				else if (land_val == STPBiomeRegistry::Jungle.ID) {
					l = STPBiomeRegistry::JungleHills.ID;
				}
				else if (land_val == STPBiomeRegistry::Savannah.ID) {
					l = STPBiomeRegistry::SavannahPlateau.ID;
				}
				else if (land_val == STPBiomeRegistry::Swamp.ID) {
					l = STPBiomeRegistry::SwampHills.ID;
				}
				else if (land_val == STPBiomeRegistry::Badlands.ID) {
					l = STPBiomeRegistry::BadlandsPlateau.ID;
				}
				//randomly generate some deep ocean as hills
				else if (land_val == STPBiomeRegistry::Ocean.ID) {
					l = STPBiomeRegistry::DeepOcean.ID;
				}
				else if (land_val == STPBiomeRegistry::WarmOcean.ID) {
					l = STPBiomeRegistry::DeepWarmOcean.ID;
				}
				else if (land_val == STPBiomeRegistry::LukewarmOcean.ID) {
					l = STPBiomeRegistry::DeepLukewarmOcean.ID;
				}
				else if (land_val == STPBiomeRegistry::ColdOcean.ID) {
					l = STPBiomeRegistry::DeepColdOcean.ID;
				}
				else if (land_val == STPBiomeRegistry::FrozenOcean.ID) {
					l = STPBiomeRegistry::DeepFrozenOcean.ID;
				}

				//now let's add some island at the centre of some ocean, given 1/3 chance of spawning
				if (STPBiomeRegistry::isOcean(land_val) && !STPBiomeRegistry::isShallowOcean(land_val) && rng.nextValue(3) == 0) {
					//filter out deep ocean
					//giving 1/2 chance of each biome, feel free to add some more...
					l = rng.nextValue(2) == 0 ? STPBiomeRegistry::Plains.ID : STPBiomeRegistry::Forest.ID;
				}

				//make sure the hill is strictly at the centre of the biome, not on the edge
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