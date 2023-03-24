#pragma once
#ifndef _STP_CLIMATE_LAYER_H_
#define _STP_CLIMATE_LAYER_H_

#include "STPCrossLayer.h"

namespace {

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
		class STPClimateSingle : public STPLayer {
		public:

			STPClimateSingle(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer* const parent) :
				STPLayer(cache_size, global_seed, salt, { parent }) {

			}

			Sample sample(const int x, const int y, const int z) override {
				//get the sample from the previous layer
				const Sample val = this->getAscendant().retrieve(x, y, z);
				//get the local RNG
				const STPLayer::STPLocalSampler rng = this->createLocalSampler(x, z);

				if (Reg::isShallowOcean(val)) {
					//if it's ocean, we don't touch it
					return val;
				}

				//generate land portion
				//1/6 chance of getting a forest or mountain
				const Sample i = rng.nextValue(6);
				switch (i) {
				case 0u: return Reg::Forest.ID;
					break;
				case 1u: return Reg::Mountain.ID;
					break;
				//otherwise it's still a plains
				default: return Reg::Plains.ID;
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
				if (center == Reg::Plains.ID
					&& (!Reg::applyAll([](Sample val) -> bool {
						return !(val == Reg::Mountain.ID || val == Reg::Forest.ID);
						}, north, east, south, west))) {
					//land
					return Reg::Desert.ID;
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
				if (center != Reg::Forest.ID
					|| Reg::applyAll([](Sample val) -> bool {
						return val != Reg::Plains.ID && val != Reg::Desert.ID;
						}, north, east, south, west)) {
					//land
					return center;
				}

				return Reg::Mountain.ID;
			}

		};

	};

}
#endif//_STP_CLIMATE_LAYER_H_