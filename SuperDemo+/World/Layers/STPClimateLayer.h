#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "STPCrossLayer.h"
#include "../Biomes/STPBiomeRegistry.h"

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPClimateLayer starts to populate the map with climate (temperature and precipitation/humidity) for later generation.
	 * In this layer(s), biome numbers are interpreted as (just an interpretation, not the actual biome):
	 * plains - hot or wet
	 * desert - temperate or humid
	 * mountains - cool or moderate
	 * forest - cold or dry
	 * Additionally ocean temperatures are also calculated in this layer
	*/
	struct STPClimateLayer {
	private:

		//Do not init this class separately
		STPClimateLayer() = default;

		~STPClimateLayer() = default;

	public:

		/**
		 * @brief STPClimateSingle adds only cold or dry climate
		*/
		class STPClimateSingle : public SuperTerrainPlus::STPDiversity::STPLayer {
		public:

			STPClimateSingle(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer* const parent) :
				STPLayer(cache_size, global_seed, salt, parent) {

			}

			Sample sample(const int x, const int y, const int z) override {
				//get the sample from the previous layer
				const Sample val = this->getAscendant()->retrieve(x, y, z);
				//get the local RNG
				const STPLayer::STPLocalSampler rng = this->createLocalSampler(x, z);

				if (STPBiomeRegistry::isShallowOcean(val)) {
					//if it's ocean, we don't touch it
					return val;
				}

				//generate land portion
				//1/6 chance of getting a forest or mountain
				const Sample i = rng.nextValue(6);
				switch (i) {
				case 0u: return STPBiomeRegistry::Forest.ID;
					break;
				case 1u: return STPBiomeRegistry::Mountain.ID;
					break;
				//otherwise it's still a plains
				default: return STPBiomeRegistry::Plains.ID;
					break;
				}

			}

		};

		/**
		 * @brief STPClimateModerate adds climate with closed temperature and humidity, e.g., cool and warm, humid and moderate
		*/
		class STPClimateModerate : public STPCrossLayer {
		public:

			STPClimateModerate(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer* const parent) :
				STPCrossLayer(cache_size, global_seed, salt, parent) {

			}

			Sample sample(const Sample center, const Sample north, const Sample east, const Sample south,
				const Sample west, Seed) override {
				//escape the one that is extreme on the centre
				//and either temperate on one of the other side
				//and replace it with a more temperate biome
				if (center == STPBiomeRegistry::Plains.ID
					&& (!STPBiomeRegistry::applyAll([](Sample val) -> bool {
						return !(val == STPBiomeRegistry::Mountain.ID || val == STPBiomeRegistry::Forest.ID);
						}, north, east, south, west))) {
					//land
					return STPBiomeRegistry::Desert.ID;
				}

				return center;
			}

		};

		/**
		 * @brief STPClimateExtreme adds climate with extreme temperature and humidity, e.g., hot and cold, wet and dry
		*/
		class STPClimateExtreme : public STPCrossLayer {
		public:

			STPClimateExtreme(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer* const parent) :
				STPCrossLayer(cache_size, global_seed, salt, parent) {

			}

			Sample sample(const Sample center, const Sample north, const Sample east, const Sample south,
				const Sample west, Seed) override {
				//escape the one that is cold on the centre
				//and either hot or warm on one of the other side
				//extreme climate cannot be placed together
				if (center != STPBiomeRegistry::Forest.ID
					|| STPBiomeRegistry::applyAll([](Sample val) -> bool {
						return val != STPBiomeRegistry::Plains.ID && val != STPBiomeRegistry::Desert.ID;
						}, north, east, south, west)) {
					//land
					return center;
				}

				return STPBiomeRegistry::Mountain.ID;
			}

		};

	};

}
#endif//_STP_LAYERS_ALL_HPP_
