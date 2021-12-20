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

			STPClimateSingle(Seed global_seed, Seed salt, STPLayer* parent) : STPLayer(global_seed, salt, parent) {

			}

			Sample sample(int x, int y, int z) override {
				//get the sample from the previous layer
				const Sample val = this->getAscendant()->retrieve(x, y, z);
				//set the local seed
				const Seed local_seed = this->genLocalSeed(x, z);
				//get the local rng
				const STPLayer::STPLocalRNG rng = this->getRNG(local_seed);

				if (STPBiomeRegistry::isShallowOcean(val)) {
					//if it's ocean, we don't touch it
					return val;
				}

				//generate land portion
				//1/6 chance of getting a forest or mountain
				const Sample i = rng.nextVal(6);
				switch (i) {
				case 0u: return STPBiomeRegistry::FOREST.getID();
					break;
				case 1u: return STPBiomeRegistry::MOUNTAIN.getID();
					break;
				//otherwise it's still a plains
				default: return STPBiomeRegistry::PLAINS.getID();
					break;
				}

			}

		};

		/**
		 * @brief STPClimateModerate adds climate with closed temperature and humidity, e.g., cool and warm, humid and moderate
		*/
		class STPClimateModerate : public STPCrossLayer {
		public:

			STPClimateModerate(Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(global_seed, salt, parent) {

			}

			Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed) override {
				//escape the one that is extreme on the center
				//and either temperate on one of the other side
				//and replace it with a more temperate biome
				if (center == STPBiomeRegistry::PLAINS.getID()
					&& (!STPBiomeRegistry::applyAll([](Sample val) -> bool {
						return !(val == STPBiomeRegistry::MOUNTAIN.getID() || val == STPBiomeRegistry::FOREST.getID());
						}, north, east, south, west))) {
					//land
					return STPBiomeRegistry::DESERT.getID();
				}

				return center;
			}

		};

		/**
		 * @brief STPClimateExtreme adds climate with extreme temperature and humidity, e.g., hot and cold, wet and dry
		*/
		class STPClimateExtreme : public STPCrossLayer {
		public:

			STPClimateExtreme(Seed global_seed, Seed salt, STPLayer* parent) : STPCrossLayer(global_seed, salt, parent) {

			}

			Sample sample(Sample center, Sample north, Sample east, Sample south, Sample west, Seed) override {
				//escape the one that is cold on the center
				//and either hot or warm on one of the other side
				//extreme climate cannot be placed together
				if (center != STPBiomeRegistry::FOREST.getID()
					|| STPBiomeRegistry::applyAll([](Sample val) -> bool {
						return val != STPBiomeRegistry::PLAINS.getID() && val != STPBiomeRegistry::DESERT.getID();
						}, north, east, south, west)) {
					//land
					return center;
				}

				return STPBiomeRegistry::MOUNTAIN.getID();
			}

		};

	};

}
#endif//_STP_LAYERS_ALL_HPP_
