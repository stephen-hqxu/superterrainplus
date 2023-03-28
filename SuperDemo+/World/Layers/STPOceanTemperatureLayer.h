#pragma once
#ifndef _STP_OCEAN_TEMPERATURE_LAYER_H_
#define _STP_OCEAN_TEMPERATURE_LAYER_H_

#include "STPCrossLayer.h"

namespace {

	/**
	 * @brief STPOceanTemperatureLayer generates different temperature region for ocean biomes and smooth the transition as much as possible
	*/
	class STPOceanTemperatureLayer {
	public:

		/**
		 * @brief STPOceanNoise setup warm and frozen ocean first with RNG
		*/
		class STPOceanNoise : public STPLayer {
		public:

			STPOceanNoise(const size_t cache_size, const STPSeed_t global_seed, const STPSeed_t salt) :
				STPLayer(cache_size, global_seed, salt) {

			}

			STPSample_t sample(const int x, int, const int z) override {
				//get local RNG
				const STPLayer::STPLocalSampler rng = this->createLocalSampler(x, z);

				//the RNG will overwrite land portion and mix them later in transition layer
				//given 1/3 chance for each temp
				const STPSample_t i = rng.nextValue(3);
				switch (i) {
				case 0u: return Reg::FrozenOcean.ID;
					break;
				case 1u: return Reg::WarmOcean.ID;
					break;
				default: return Reg::Ocean.ID;
					break;
				}
			}

		};

		/**
		 * @brief STPOceanTemperate adds smooth transition between lukewarm and cold
		*/
		class STPOceanTemperate : public STPCrossLayer {
		public:

			STPOceanTemperate(const size_t cache_size, const STPSeed_t global_seed, const STPSeed_t salt, STPLayer& parent) :
				STPCrossLayer(cache_size, global_seed, salt, parent) {
				//parent: STPOceanExtreme
			}

			STPSample_t sample(const STPSample_t center, const STPSample_t north, const STPSample_t east, const STPSample_t south,
				const STPSample_t west, STPSeed_t) override {
				if (center != Reg::LukewarmOcean.ID
					|| Reg::applyAll([](STPSample_t val) -> bool {
						return val != Reg::ColdOcean.ID;
						}, north, east, south, west)) {
					return center;
				}

				//or cold
				if (center != Reg::ColdOcean.ID
					|| Reg::applyAll([](STPSample_t val) -> bool {
						return val != Reg::LukewarmOcean.ID;
						}, north, east, south, west)) {
					return center;
				}

				//lukewarm meets cold in either order = ocean
				return Reg::Ocean.ID;
			}

		};

		/**
		 * @brief STPOceanExtreme adds smooth transition between warm and frozen ocean
		*/
		class STPOceanExtreme : public STPCrossLayer {
		public:

			STPOceanExtreme(const size_t cache_size, const STPSeed_t global_seed, const STPSeed_t salt, STPLayer& parent) :
				STPCrossLayer(cache_size, global_seed, salt, parent) {
				//parent: STPOceanNoise
			}

			STPSample_t sample(const STPSample_t center, const STPSample_t north, const STPSample_t east,
				const STPSample_t south, const STPSample_t west, STPSeed_t) override {
				if (center == Reg::WarmOcean.ID
					&& (!Reg::applyAll([](STPSample_t val) -> bool {
						return val != Reg::FrozenOcean.ID;
						}, north, east, south, west))) {
					//warm meets frozen = lukewarm
					return Reg::LukewarmOcean.ID;
				}

				//or frozen, it can only be either of both, then vice versa
				if (center == Reg::FrozenOcean.ID
					&& (!Reg::applyAll([](STPSample_t val) -> bool {
						return val != Reg::WarmOcean.ID;
						}, north, east, south, west))) {
					//frozen meets warm = cold
					return Reg::ColdOcean.ID;
				}

				//otherwise do nothing
				return center;
			}
		};

		/**
		 * @brief STPOceanTransition smooths out the temp of the ocean when it meets land
		*/
		class STPOceanTransition : public STPLayer {
		public:

			STPOceanTransition(
				const size_t cache_size, const STPSeed_t global_seed, const STPSeed_t salt, STPLayer& parent) :
				STPLayer(cache_size, global_seed, salt, { parent }) {

			}

			STPSample_t sample(const int x, const int y, const int z) override {
				//get the value from previous layer
				const STPSample_t val = this->getAscendant().retrieve(x, y, z);
				//don't touch it if it's land
				if (!Reg::isOcean(val)) {
					return val;
				}

				//testing for neighbours and check for lands
				for (int rx = -8; rx <= 8; rx += 4) {
					for (int rz = -8; rz <= 8; rz += 4) {
						const STPSample_t shift_xz = this->getAscendant().retrieve(x + rx, y, z + rz);
						if (Reg::isOcean(shift_xz)) {
							//we need to find neighbour who is land
							continue;
						}

						if (val == Reg::WarmOcean.ID) {
							return Reg::LukewarmOcean.ID;
						}
						if (val == Reg::FrozenOcean.ID) {
							return Reg::ColdOcean.ID;
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
		class STPOceanMix : public STPLayer {
		public:

			STPOceanMix(const size_t cache_size, const STPSeed_t global_seed, const STPSeed_t salt, STPLayer& land, STPLayer& ocean) :
				STPLayer(cache_size, global_seed, salt, { land, ocean }) {
				//parent 0: land
				//parent 1: STPOceanTemperate
			}

			STPSample_t sample(const int x, const int y, const int z) override {
				//get the land value from the land layer
				const STPSample_t land = this->getAscendant(0).retrieve(x, y, z);
				//don't touch it if it's land
				if (!Reg::isOcean(land)) {
					return land;
				}

				//otherwise return the respective ocean section
				return this->getAscendant(1).retrieve(x, y, z);
			}

		};

	};

}
#endif//_STP_OCEAN_TEMPERATURE_LAYER_H_