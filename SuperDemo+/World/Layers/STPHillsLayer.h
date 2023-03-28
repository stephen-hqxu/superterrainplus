#pragma once
#ifndef _STP_HILLS_LAYER_H_
#define _STP_HILLS_LAYER_H_

namespace {

	/**
	 * @brief STPHillsLayer generates hills that are located at the central of some biomes from the noise function
	*/
	class STPHillsLayer : public STPLayer {
	private:

		//Chance of having a hill
		static constexpr STPSample_t HillsChance = 29u;

	public:

		STPHillsLayer(const size_t cache_size, const STPSeed_t global_seed, const STPSeed_t salt, STPLayer& land,
			STPLayer& noise) : STPLayer(cache_size, global_seed, salt, { land, noise }) {
			//parent 0: land / biome
			//parent 1: noise
		}

		STPSample_t sample(const int x, const int y, const int z) override {
			//get the local RNG
			const STPLayer::STPLocalSampler rng = this->createLocalSampler(x, z);
			//get the parent samples
			const STPSample_t land_val = this->getAscendant(0).retrieve(x, y, z);
			const STPSample_t noise_val = this->getAscendant(1).retrieve(x, y, z);
			
			//chance of having a hill
			const STPSample_t is_hill = (noise_val - 2) % STPHillsLayer::HillsChance;

			//1/3 chance to have a hill
			if (rng.nextValue(3) == 0 || is_hill == 0) {
				STPSample_t l = land_val;
				//convert biomes to their respective hill biome
				if (land_val == Reg::Desert.ID) {
					l = Reg::DesertHills.ID;
				}
				else if (land_val == Reg::Taiga.ID) {
					l = Reg::TaigaHills.ID;
				}
				else if (land_val == Reg::Mountain.ID) {
					l = Reg::WoodedMountain.ID;
				}
				else if (land_val == Reg::SnowyTundra.ID || land_val == Reg::SnowyTaiga.ID) {
					l = Reg::SnowyMountain.ID;
				}
				else if (land_val == Reg::Plains.ID) {
					l = rng.nextValue(3) == 0 ? Reg::ForestHills.ID : Reg::Forest.ID;
				}
				else if (land_val == Reg::Forest.ID) {
					l = Reg::ForestHills.ID;
				}
				else if (land_val == Reg::Jungle.ID) {
					l = Reg::JungleHills.ID;
				}
				else if (land_val == Reg::Savannah.ID) {
					l = Reg::SavannahPlateau.ID;
				}
				else if (land_val == Reg::Swamp.ID) {
					l = Reg::SwampHills.ID;
				}
				else if (land_val == Reg::Badlands.ID) {
					l = Reg::BadlandsPlateau.ID;
				}
				//randomly generate some deep ocean as hills
				else if (land_val == Reg::Ocean.ID) {
					l = Reg::DeepOcean.ID;
				}
				else if (land_val == Reg::WarmOcean.ID) {
					l = Reg::DeepWarmOcean.ID;
				}
				else if (land_val == Reg::LukewarmOcean.ID) {
					l = Reg::DeepLukewarmOcean.ID;
				}
				else if (land_val == Reg::ColdOcean.ID) {
					l = Reg::DeepColdOcean.ID;
				}
				else if (land_val == Reg::FrozenOcean.ID) {
					l = Reg::DeepFrozenOcean.ID;
				}

				//now let's add some island at the centre of some ocean, given 1/3 chance of spawning
				if (Reg::isOcean(land_val) && !Reg::isShallowOcean(land_val) && rng.nextValue(3) == 0) {
					//filter out deep ocean
					//giving 1/2 chance of each biome, feel free to add some more...
					l = rng.nextValue(2) == 0 ? Reg::Plains.ID : Reg::Forest.ID;
				}

				//make sure the hill is strictly at the centre of the biome, not on the edge
				if (l != land_val) {
					unsigned char m = 0x00u;
					if (land_val ==	this->getAscendant(0).retrieve(x, y, z - 1)) {
						m++;
					}
					if (land_val == this->getAscendant(0).retrieve(x + 1, y, z)) {
						m++;
					}
					if (land_val == this->getAscendant(0).retrieve(x - 1, y, z)) {
						m++;
					}
					if (land_val == this->getAscendant(0).retrieve(x, y, z + 1)) {
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
#endif//_STP_HILLS_LAYER_H_