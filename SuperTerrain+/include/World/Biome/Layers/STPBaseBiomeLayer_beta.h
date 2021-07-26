#pragma once
#ifdef _STP_LAYERS_ALL_HPP_

#include "../STPLayer.h"
#include "../Biomes/STPBiomeRegistry.h"

#include <assert.h>

#include <vector>
#include <algorithm>

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPBiome::Seed;
	using SuperTerrainPlus::STPBiome::Sample;

	/**
	 * @brief STPBaseBiomeLayer starts to add biomes based on the climate, and interprete temperature and precipitation to the actual biome
	*/
	class STPBaseBiomeLayer : public SuperTerrainPlus::STPBiome::STPLayer {
	private:

		typedef std::vector<Sample> BiomeList;
		//Classification of row - cold, cool, temperate/warm, hot
		//of column - (dry, moderate), (humid, wet)
		typedef BiomeList ClimateMatrix[4][2];

		//A table of interpretation
		//Since biome ids are loaded after runtime, so we can't make it static
		//access using [row][column], row is temp, column is prec, return an array of satisfied biomes
		ClimateMatrix ClimateClassification;

		//When there is no biome for this climate, we choose one from this list
		const BiomeList DEFAULT_BIOMES = {
			STPBiomeRegistry::PLAINS.getID(),
			STPBiomeRegistry::MOUNTAIN.getID(),
			STPBiomeRegistry::FOREST.getID()
		};

		//The thresholds of temp and precipitation
		static constexpr float TRESHOLDS[4] = {1.0f, 1.5f, 2.5f, 3.0f};

		void addClimatables(BiomeList& climatable) {
			auto climate_insert = [&climatable](const STPBiome& biome) -> void {
				climatable.push_back(biome.getID());
				return;
			};
			//Only regular, vanilla, non-variant land biomes are allowed to be climatable biome
			//other biomes will be generated in different layer with different function
			//basically all land biomes are fine
			climate_insert(STPBiomeRegistry::DESERT);
			climate_insert(STPBiomeRegistry::MOUNTAIN);
			climate_insert(STPBiomeRegistry::FOREST);
			climate_insert(STPBiomeRegistry::TAIGA);
			climate_insert(STPBiomeRegistry::SNOWY_TAIGA);
			climate_insert(STPBiomeRegistry::SNOWY_TUNDRA);
			climate_insert(STPBiomeRegistry::JUNGLE);
			climate_insert(STPBiomeRegistry::SAVANA);
			climate_insert(STPBiomeRegistry::SWAMP);
			climate_insert(STPBiomeRegistry::BADLANDS);
		}

		void addExplicit() {
			//adding explicit biomes
			auto insert_explicit = [&c = this->ClimateClassification](unsigned short temp_class, unsigned short prec_class, const STPBiome& biome) -> void {
				auto& biome_list = c[temp_class][prec_class >> 1u];

				if (std::find(biome_list.begin(), biome_list.end(), biome.getID()) != biome_list.end()) {
					//biome does not exist, we can add
					biome_list.push_back(biome.getID());
				}
				return;
			};
			//to give a higher chance of spawning
			insert_explicit(1, 1, STPBiomeRegistry::PLAINS);
			insert_explicit(1, 2, STPBiomeRegistry::PLAINS);
			insert_explicit(2, 1, STPBiomeRegistry::PLAINS);
			insert_explicit(2, 2, STPBiomeRegistry::PLAINS);
			insert_explicit(1, 1, STPBiomeRegistry::FOREST);
			insert_explicit(1, 2, STPBiomeRegistry::FOREST);
			insert_explicit(2, 1, STPBiomeRegistry::FOREST);
			insert_explicit(2, 2, STPBiomeRegistry::FOREST);
			insert_explicit(1, 1, STPBiomeRegistry::MOUNTAIN);
			insert_explicit(1, 2, STPBiomeRegistry::MOUNTAIN);
			insert_explicit(2, 1, STPBiomeRegistry::MOUNTAIN);
			insert_explicit(2, 2, STPBiomeRegistry::MOUNTAIN);
			insert_explicit(2, 3, STPBiomeRegistry::SWAMP);
			insert_explicit(3, 3, STPBiomeRegistry::SWAMP);
			insert_explicit(2, 2, STPBiomeRegistry::JUNGLE);
			insert_explicit(2, 3, STPBiomeRegistry::JUNGLE);
			insert_explicit(3, 2, STPBiomeRegistry::JUNGLE);
			insert_explicit(3, 3, STPBiomeRegistry::JUNGLE);
		}

		void initClasses() {
			//Biomes that can be a base climate biome
			BiomeList climatable_biome;

			//Adding climatable biomes
			this->addClimatables(climatable_biome);

			//when the biome is valid for climate
			for (auto applicable : climatable_biome) {
				const auto biome = STPBiomeRegistry::REGISTRY.find(applicable);

				//determine which temp and prec category is
				short index[2] = {-1, -1};
				for (short i = 0; i < 4 && (index[0] == -1 || index[1] == -1); i++) { 
					if (index[0] != -1 &&  biome->second->getTemperature() <= STPBaseBiomeLayer::TRESHOLDS[i]) {
						index[0] = i;
					}

					if (index[1] != -1 && biome->second->getPrecipitation() <= STPBaseBiomeLayer::TRESHOLDS[i]) {
						index[1] = i;
					}
				}

				if (index[0] != -1 && index[1] != -1) {
					//category found, insert into matrix
					this->ClimateClassification[index[0]][index[1] >> 1].push_back(biome->second->getID());
				}
			}

			this->addExplicit();
		}

		/**
		 * @brief Interpret the sample from the previous and convert into temp or prec
		 * @param sample The sample, either from temp map or prec map
		 * @return The interpreted climate category, can be used as an index to the lookup matrix
		*/
		static unsigned short interpret(Sample sample) {
			if (sample == STPBiomeRegistry::PLAINS.getID()) {
				return 3u;
			}
			if (sample == STPBiomeRegistry::DESERT.getID()) {
				return 2u;
			}
			if (sample == STPBiomeRegistry::MOUNTAIN.getID()) {
				return 1u;
			}
			if (sample == STPBiomeRegistry::FOREST.getID()) {
				return 0u;
			}

			return 0u;
		}

	public:

		STPBaseBiomeLayer(Seed global_seed, Seed salt, STPLayer* parent_temp, STPLayer* parent_prec) : STPLayer(global_seed, salt, parent_temp, parent_prec) {
			//parent 0: temperature map
			//parent 1: precipitation map
			this->initClasses();
		}

		Sample sample(int x, int y, int z) override {
			//set local seed
			const Seed local_seed = this->genLocalSeed(x, z);
			//get local rng
			const STPLayer::STPLocalRNG rng = this->getRNG(local_seed);

			//get temp and prec for this coordinate
			const Sample temp = this->getAscendant(0)->sample_cached(x, y, z);
			const Sample prec = this->getAscendant(1)->sample_cached(x, y, z);
			//get the index of lookup table
			const unsigned short row = STPBaseBiomeLayer::interpret(temp);
			const unsigned short column = STPBaseBiomeLayer::interpret(prec);
			
			//randomly choose a biome that satisfies our temp and prec
			const BiomeList& biomes = this->ClimateClassification[row][column >> 1u];

			//there is no satifying biome, randomly choose one from the default biome
			return biomes.empty() ? this->DEFAULT_BIOMES[rng.nextVal(this->DEFAULT_BIOMES.size())] : biomes[rng.nextVal(biomes.size())];
		}

	};

}
#endif//_STP_LAYERS_ALL_HPP_