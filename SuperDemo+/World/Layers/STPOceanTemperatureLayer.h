#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "STPCrossLayer.h"
#include "../Biomes/STPBiomeRegistry.h"

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPOceanTemperatureLayer generates different temperature region for ocean biomes and smooth the transition as much as possible
	*/
	class STPOceanTemperatureLayer {
	public:

		/**
		 * @brief STPOceanNoise setups warm and frozen ocean first with RNG
		*/
		class STPOceanNoise : public SuperTerrainPlus::STPDiversity::STPLayer {
		public:

			STPOceanNoise(Seed global_seed, Seed salt) : STPLayer(global_seed, salt) {

			}

			Sample sample(int x, int, int z) override {
				//set local seed
				const Seed local_seed = this->genLocalSeed(x, z);
				//get local rng
				const STPLayer::STPLocalRNG rng = this->getRNG(local_seed);

				//the rng will overwrite land portion and mix them later in transition layer
				//given 1/3 chance for each temp
				const Sample i = rng.nextVal(3);
				switch (i) {
				case 0u: return STPBiomeRegistry::FROZEN_OCEAN.getID();
					break;
				case 1u: return STPBiomeRegistry::WARM_OCEAN.getID();
					break;
				default: return STPBiomeRegistry::OCEAN.getID();
					break;
				}
			}

		};

		/**
		 * @brief STPOceanTemperate adds smooth transition between lukewarm and cold
		*/
		class STPOceanTemperate : public STPCrossLayer {
		public:

			STPOceanTemperate(Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(global_seed, salt, parent) {
				//parent: STPOceanExtreme
			}

			Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed) override {
				if (center != STPBiomeRegistry::LUKEWARM_OCEAN.getID()
					|| STPBiomeRegistry::applyAll([](Sample val) -> bool {
						return val != STPBiomeRegistry::COLD_OCEAN.getID();
						}, north, east, south, west)) {
					return center;
				}

				//or cold
				if (center != STPBiomeRegistry::COLD_OCEAN.getID()
					|| STPBiomeRegistry::applyAll([](Sample val) -> bool {
						return val != STPBiomeRegistry::LUKEWARM_OCEAN.getID();
						}, north, east, south, west)) {
					return center;
				}

				//lukewarm meets cold in either order = ocean
				return STPBiomeRegistry::OCEAN.getID();
			}

		};

		/**
		 * @brief STPOceanExtreme adds smooth transition between warm and frozen ocean
		*/
		class STPOceanExtreme : public STPCrossLayer {
		public:

			STPOceanExtreme(Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(global_seed, salt, parent) {
				//parent: STPOceanNoise
			}

			Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed) override {
				if (center == STPBiomeRegistry::WARM_OCEAN.getID()
					&& (!STPBiomeRegistry::applyAll([](Sample val) -> bool {
						return val != STPBiomeRegistry::FROZEN_OCEAN.getID();
						}, north, east, south, west))) {
					//warm meets frozen = lukewarm
					return STPBiomeRegistry::LUKEWARM_OCEAN.getID();
				}

				//or frozen, it can only be either of both, then vice versa
				if (center == STPBiomeRegistry::FROZEN_OCEAN.getID()
					&& (!STPBiomeRegistry::applyAll([](Sample val) -> bool {
						return val != STPBiomeRegistry::WARM_OCEAN.getID();
						}, north, east, south, west))) {
					//frozen meets warm = cold
					return STPBiomeRegistry::COLD_OCEAN.getID();
				}

				//otherwise do nothing
				return center;
			}
		};

		/**
		 * @brief STPOceanTransition smooths out the temp of the ocean when it meets land
		*/
		class STPOceanTransition : public SuperTerrainPlus::STPDiversity::STPLayer {
		public:

			STPOceanTransition(Seed global_seed, Seed salt, STPLayer* parent) : STPLayer(global_seed, salt, parent) {

			}

			Sample sample(int x, int y, int z) override {
				//get the value from previous layer
				const Sample val = this->getAscendant()->retrieve(x, y, z);
				//don't touch it if it's land
				if (!STPBiomeRegistry::isOcean(val)) {
					return val;
				}

				//testing for neighbors and check for lands
				for (int rx = -8; rx <= 8; rx += 4) {
					for (int rz = -8; rz <= 8; rz += 4) {
						const Sample shift_xz = this->getAscendant()->retrieve(x + rx, y, z + rz);
						if (STPBiomeRegistry::isOcean(shift_xz)) {
							//we need to find neighbor who is land
							continue;
						}

						if (val == STPBiomeRegistry::WARM_OCEAN.getID()) {
							return STPBiomeRegistry::LUKEWARM_OCEAN.getID();
						}
						if (val == STPBiomeRegistry::FROZEN_OCEAN.getID()) {
							return STPBiomeRegistry::COLD_OCEAN.getID();
						}
					}
				}

				//otherwise do nothing
				return val;
			}

		};

		/**
		 * @brief STPOceanMix mixes ocean temp layers with the original land
		*/
		class STPOceanMix : public SuperTerrainPlus::STPDiversity::STPLayer {
		public:

			STPOceanMix(Seed global_seed, Seed salt, STPLayer* land, STPLayer* ocean) : STPLayer(global_seed, salt, land, ocean) {
				//parent 0: land
				//parent 1: STPOceanTemperate
			}

			Sample sample(int x, int y, int z) override {
				//get the land value from the land layer
				const Sample land = this->getAscendant(0)->retrieve(x, y, z);
				//don't touch it if it's land
				if (!STPBiomeRegistry::isOcean(land)) {
					return land;
				}

				//otherwise return the respective ocean section
				return this->getAscendant(1)->retrieve(x, y, z);
			}

		};

	};

}
#endif//_STP_LAYERS_ALL_HPP_