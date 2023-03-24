#pragma once
#ifndef _STP_BASE_BIOME_LAYER_H_
#define _STP_BASE_BIOME_LAYER_H_

#include <array>

namespace {

	/**
	 * @brief STPBaseBiomeLayer starts to add biomes based on the climate, and interpret temperature and precipitation to the actual biome
	*/
	class STPBaseBiomeLayer : public STPLayer {
	private:

		template<size_t S>
		using STPBiomeList = std::array<Sample, S>;

		//A table of interpretation
		//If we want higher chance of spawning for specific biomes, just repeat that one
		//Since biome ids are loaded after runtime, so we can't make it static

		const STPBiomeList<4u> DryBiomes = {
			Reg::Desert.ID,
			Reg::Savannah.ID,
			Reg::Plains.ID,
			Reg::Badlands.ID
		};

		const STPBiomeList<5u> TemperateBiomes = {
			Reg::Jungle.ID,
			Reg::Forest.ID,
			Reg::Mountain.ID,
			Reg::Plains.ID,
			Reg::Swamp.ID
		};

		const STPBiomeList<4u> CoolBiomes = {
			Reg::Plains.ID,
			Reg::Mountain.ID,
			Reg::Forest.ID,
			Reg::Taiga.ID
		};

		const STPBiomeList<3u> SnowyBiomes = {
			Reg::SnowyTaiga.ID,
			Reg::SnowyTundra.ID,
			Reg::SnowyMountain.ID
		};

	public:

		STPBaseBiomeLayer(const size_t cache_size, const Seed global_seed, const Seed salt, STPLayer* const parent) :
			STPLayer(cache_size, global_seed, salt, { parent }) {
			//parent:: climate layer
		}

		Sample sample(const int x, const int y, const int z) override {
			//get the local RNG
			STPLayer::STPLocalSampler rng = this->createLocalSampler(x, z);
			//get the climate for this local coordinate
			const Sample climate = this->getAscendant().retrieve(x, y, z);

			//if it's ocean, we should leave it untouched
			if (Reg::isOcean(climate)) {
				return climate;
			}
			
			//interpretation, compared to vanilla Minecraft, special climate has been removed, every biomes have the equal chance of spawning
			if (climate == Reg::Plains.ID) {
				//dry and hot biome
				return this->DryBiomes[rng.nextValue(static_cast<Sample>(this->DryBiomes.size()))];
			}
			if (climate == Reg::Desert.ID) {
				//temperate biome
				return this->TemperateBiomes[rng.nextValue(static_cast<Sample>(this->TemperateBiomes.size()))];
			}
			if (climate == Reg::Mountain.ID) {
				//cool biome
				return this->CoolBiomes[rng.nextValue(static_cast<Sample>(this->CoolBiomes.size()))];
			}
			if (climate == Reg::Forest.ID) {
				//snowy and cold biome
				return this->SnowyBiomes[rng.nextValue(static_cast<Sample>(this->SnowyBiomes.size()))];
			}

			//this usually won't happen, but just in case
			return climate;
		}

	};

}
#endif//_STP_BASE_BIOME_LAYER_H_