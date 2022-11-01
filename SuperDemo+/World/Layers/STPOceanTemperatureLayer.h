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
		 * @brief STPOceanNoise setup warm and frozen ocean first with RNG
		*/
		class STPOceanNoise : public SuperTerrainPlus::STPDiversity::STPLayer {
		public:

			STPOceanNoise(size_t cache_size, Seed global_seed, Seed salt) : STPLayer(cache_size, global_seed, salt) {

			}

			Sample sample(int x, int, int z) override {
				//get local RNG
				const STPLayer::STPLocalSampler rng = this->createLocalSampler(x, z);

				//the RNG will overwrite land portion and mix them later in transition layer
				//given 1/3 chance for each temp
				const Sample i = rng.nextValue(3);
				switch (i) {
				case 0u: return STPBiomeRegistry::FrozenOcean.ID;
					break;
				case 1u: return STPBiomeRegistry::WarmOcean.ID;
					break;
				default: return STPBiomeRegistry::Ocean.ID;
					break;
				}
			}

		};

		/**
		 * @brief STPOceanTemperate adds smooth transition between lukewarm and cold
		*/
		class STPOceanTemperate : public STPCrossLayer {
		public:

			STPOceanTemperate(size_t cache_size, Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(cache_size, global_seed, salt, parent) {
				//parent: STPOceanExtreme
			}

			Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed) override {
				if (center != STPBiomeRegistry::LukewarmOcean.ID
					|| STPBiomeRegistry::applyAll([](Sample val) -> bool {
						return val != STPBiomeRegistry::ColdOcean.ID;
						}, north, east, south, west)) {
					return center;
				}

				//or cold
				if (center != STPBiomeRegistry::ColdOcean.ID
					|| STPBiomeRegistry::applyAll([](Sample val) -> bool {
						return val != STPBiomeRegistry::LukewarmOcean.ID;
						}, north, east, south, west)) {
					return center;
				}

				//lukewarm meets cold in either order = ocean
				return STPBiomeRegistry::Ocean.ID;
			}

		};

		/**
		 * @brief STPOceanExtreme adds smooth transition between warm and frozen ocean
		*/
		class STPOceanExtreme : public STPCrossLayer {
		public:

			STPOceanExtreme(size_t cache_size, Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(cache_size, global_seed, salt, parent) {
				//parent: STPOceanNoise
			}

			Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed) override {
				if (center == STPBiomeRegistry::WarmOcean.ID
					&& (!STPBiomeRegistry::applyAll([](Sample val) -> bool {
						return val != STPBiomeRegistry::FrozenOcean.ID;
						}, north, east, south, west))) {
					//warm meets frozen = lukewarm
					return STPBiomeRegistry::LukewarmOcean.ID;
				}

				//or frozen, it can only be either of both, then vice versa
				if (center == STPBiomeRegistry::FrozenOcean.ID
					&& (!STPBiomeRegistry::applyAll([](Sample val) -> bool {
						return val != STPBiomeRegistry::WarmOcean.ID;
						}, north, east, south, west))) {
					//frozen meets warm = cold
					return STPBiomeRegistry::ColdOcean.ID;
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

			STPOceanTransition(size_t cache_size, Seed global_seed, Seed salt, STPLayer* parent) : STPLayer(cache_size, global_seed, salt, parent) {

			}

			Sample sample(int x, int y, int z) override {
				//get the value from previous layer
				const Sample val = this->getAscendant()->retrieve(x, y, z);
				//don't touch it if it's land
				if (!STPBiomeRegistry::isOcean(val)) {
					return val;
				}

				//testing for neighbours and check for lands
				for (int rx = -8; rx <= 8; rx += 4) {
					for (int rz = -8; rz <= 8; rz += 4) {
						const Sample shift_xz = this->getAscendant()->retrieve(x + rx, y, z + rz);
						if (STPBiomeRegistry::isOcean(shift_xz)) {
							//we need to find neighbour who is land
							continue;
						}

						if (val == STPBiomeRegistry::WarmOcean.ID) {
							return STPBiomeRegistry::LukewarmOcean.ID;
						}
						if (val == STPBiomeRegistry::FrozenOcean.ID) {
							return STPBiomeRegistry::ColdOcean.ID;
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

			STPOceanMix(size_t cache_size, Seed global_seed, Seed salt, STPLayer* land, STPLayer* ocean) :
				STPLayer(cache_size, global_seed, salt, land, ocean) {
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