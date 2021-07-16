#pragma once
#ifndef _STP_BIOME_REGISTRY_H_
#define _STP_BIOME_REGISTRY_H_

//ADT
#include <map>
#include <type_traits>
//Biome
#include "../STPBiome.h"

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Sample;
	using SuperTerrainPlus::STPDiversity::STPBiome;
	using SuperTerrainPlus::STPSettings::STPBiomeSettings;

	/**
	 * @brief STPBiomeRegistry contains all registered biome. Each biome is assigned to an unique sampling id, which uniquely identify different biome
	*/
	struct STPBiomeRegistry {
	private:

		//Just don't init the class okay? Everything is static.
		STPBiomeRegistry() = default;

		~STPBiomeRegistry() = default;

	public:

		/**
		 * @brief STPPrecipitationType states the type of precipitation
		*/
		enum class STPPrecipitationType : unsigned char {
			//It's too hot and dry such that there is no precipitation
			NONE = 0x00,
			//Normal rain
			RAIN = 0x01,
			//It's cold so it snows
			SNOW = 0x02
		};

		//A table of settings for registered biome
		inline static std::map<Sample, const STPBiome*> REGISTRY;

		//A table of registered biome id, note that biomes are unordered

		/**
		 * @brief Ocean biome, base height is very low, with water filled atop
		*/
		inline static STPBiome OCEAN;
		/**
		 * @brief Deep ocean biome, similar to ocean biome but with even lower base height
		*/
		inline static STPBiome DEEP_OCEAN;
		/**
		 * @brief Warm ocean biome, ocean located in hot biome
		*/
		inline static STPBiome WARM_OCEAN;
		/**
		 * @brief Luke warm ocean biome, ocean located in moderate biome
		*/
		inline static STPBiome LUKEWARM_OCEAN;
		/**
		 * @brief Cold ocean biome, ocean located in cold biome
		*/
		inline static STPBiome COLD_OCEAN;
		/**
		 * @brief Frozen ocean biome, similar to ocean but water gets frozen in cold biomes or near cold biomes.
		*/
		inline static STPBiome FROZEN_OCEAN;
		/**
		 * @brief Deep warm ocean biome, ocean located in hot biome with decreased base height
		*/
		inline static STPBiome DEEP_WARM_OCEAN;
		/**
		 * @brief Deep luke warm ocean biome, ocean located in moderate biome with decreased base height
		*/
		inline static STPBiome DEEP_LUKEWARM_OCEAN;
		/**
		 * @brief Deep cold ocean biome, ocean located in cold biome with decreased base height
		*/
		inline static STPBiome DEEP_COLD_OCEAN;
		/**
		 * @brief Deep frozen ocean biome, ocean located in super cold biome with decreased base height
		*/
		inline static STPBiome DEEP_FROZEN_OCEAN;
		/**
		 * @brief Plains biome, a flat ground with little height variation, moderate temp and humidity
		*/
		inline static STPBiome PLAINS;
		/**
		 * @brief Desert biome, everything is sand, it's super hot and dry AF
		*/
		inline static STPBiome DESERT;
		/**
		 * @brief Desert hills biome, located inside the desert with higher variation
		*/
		inline static STPBiome DESERT_HILLS;
		/**
		 * @brief Mountain biome, base height is higher than most other biomes, with huge variation
		*/
		inline static STPBiome MOUNTAIN;
		/**
		 * @brief Wooded mountain biome, located inside mountain but the groud is greener, elvation is pretty much the same
		*/
		inline static STPBiome WOODED_MOUNTAIN;
		/**
		 * @brief Snowy mountatin biome, located inside snowy tundra but with higher variation
		*/
		inline static STPBiome SNOWY_MOUNTAIN;
		/**
		 * @brief Forest biome, trees everywhere, warm and humid
		*/
		inline static STPBiome FOREST;
		/**
		 * @brief Forest hills biome, located inside forest, but with higher variation
		*/
		inline static STPBiome FOREST_HILLS;
		/**
		 * @brief Taiga biome, mostly resemble a plain, but it's cold and wet.
		*/
		inline static STPBiome TAIGA;
		/**
		 * @brief Taiga gills biome, located inside taiga, but with higher variation
		*/
		inline static STPBiome TAIGA_HILLS;
		/**
		 * @brief Snowy taiga biome, similar to taiga but it's colder
		*/
		inline static STPBiome SNOWY_TAIGA;
		/**
		 * @brief Snowy tundra biome, like taiga, but it's much colder and dryer, with less vegetations.
		*/
		inline static STPBiome SNOWY_TUNDRA;
		/**
		 * @brief River biome, one of the most special biome, it goes across the map randomly, and needs to be generated with separate algorithm
		*/
		inline static STPBiome RIVER;
		/**
		 * @brief Frozen river biome, similar to river but water gets frozen in cold biomes.
		*/
		inline static STPBiome FROZEN_RIVER;
		/**
		 * @brief Beach biome, one of the edge biome system, it connects various biomes with ocean biome.
		*/
		inline static STPBiome BEACH;
		/**
		 * @brief Snowy beach biome, similar to beach biome that acts as a connection between ocean and other biomes, but it apperas in cold and snowy area.
		*/
		inline static STPBiome SNOWY_BEACH;
		/**
		 * @brief Stone shore biome, similar to beach it can be found near the ocean, but it connects with mountain only
		*/
		inline static STPBiome STONE_SHORE;
		/**
		 * @brief Jungle biome, there are a lot of trees. It's super hot and wet
		*/
		inline static STPBiome JUNGLE;
		/**
		 * @brief Jungle hills biome, located inside jungle, but with higher variation
		*/
		inline static STPBiome JUNGLE_HILLS;
		/**
		 * @brief Savana biome, basically like a plain, but it's hot and dry.
		*/
		inline static STPBiome SAVANA;
		/**
		 * @brief Savana plateau biome, located inside savana, but with higher variation
		*/
		inline static STPBiome SAVANA_PLATEAU;
		/**
		 * @brief Swamp biome, it's usually hot and very dry, with low base height so there is a lot of water filling up, mostly found inside or near jungle.
		*/
		inline static STPBiome SWAMP;
		/**
		 * @brief Swamp hills biome, located inside swamp, but with higher variation
		*/
		inline static STPBiome SWAMP_HILLS;
		/**
		 * @brief Badlands biome, a biome that is super dry and full of harden clay and rock, and eroded
		*/
		inline static STPBiome BADLANDS;
		/**
		 * @brief Badlands plateau biome, similar to badlands, but with higher variation
		*/
		inline static STPBiome BADLANDS_PLATEAU;

		//Definitions of some biome utility functions

		/**
		 * @brief Call this function to register all biomes and fill up the biome registry
		*/
		static void registerBiomes() {
			static bool initialised = false;
			if (initialised) {
				//do not re-initialise those biomes
				return;
			}

			//add all biomes to registry
			static auto reg_insert = [](const STPBiome& biome) -> void {
				STPBiomeRegistry::REGISTRY.emplace(biome.getID(), &biome);
				return;
			};
			//Oceans
			reg_insert(STPBiomeRegistry::OCEAN);
			reg_insert(STPBiomeRegistry::DEEP_OCEAN);
			reg_insert(STPBiomeRegistry::WARM_OCEAN);
			reg_insert(STPBiomeRegistry::LUKEWARM_OCEAN);
			reg_insert(STPBiomeRegistry::COLD_OCEAN);
			reg_insert(STPBiomeRegistry::FROZEN_OCEAN);
			reg_insert(STPBiomeRegistry::DEEP_WARM_OCEAN);
			reg_insert(STPBiomeRegistry::DEEP_LUKEWARM_OCEAN);
			reg_insert(STPBiomeRegistry::DEEP_COLD_OCEAN);
			reg_insert(STPBiomeRegistry::DEEP_FROZEN_OCEAN);
			//Rivers
			reg_insert(STPBiomeRegistry::RIVER);
			reg_insert(STPBiomeRegistry::FROZEN_RIVER);
			//Lands
			reg_insert(STPBiomeRegistry::PLAINS);
			reg_insert(STPBiomeRegistry::DESERT);
			reg_insert(STPBiomeRegistry::MOUNTAIN);
			reg_insert(STPBiomeRegistry::FOREST);
			reg_insert(STPBiomeRegistry::TAIGA);
			reg_insert(STPBiomeRegistry::SNOWY_TAIGA);
			reg_insert(STPBiomeRegistry::SNOWY_TUNDRA);
			reg_insert(STPBiomeRegistry::JUNGLE);
			reg_insert(STPBiomeRegistry::SAVANA);
			reg_insert(STPBiomeRegistry::SWAMP);
			reg_insert(STPBiomeRegistry::BADLANDS);
			//Hills
			reg_insert(STPBiomeRegistry::DESERT_HILLS);
			reg_insert(STPBiomeRegistry::TAIGA_HILLS);
			reg_insert(STPBiomeRegistry::WOODED_MOUNTAIN);
			reg_insert(STPBiomeRegistry::SNOWY_MOUNTAIN);
			reg_insert(STPBiomeRegistry::FOREST_HILLS);
			reg_insert(STPBiomeRegistry::JUNGLE_HILLS);
			reg_insert(STPBiomeRegistry::SAVANA_PLATEAU);
			reg_insert(STPBiomeRegistry::SWAMP_HILLS);
			reg_insert(STPBiomeRegistry::BADLANDS_PLATEAU);
			//Edges and Shores
			reg_insert(STPBiomeRegistry::BEACH);
			reg_insert(STPBiomeRegistry::SNOWY_BEACH);
			reg_insert(STPBiomeRegistry::STONE_SHORE);

			initialised = true;
		}
		
		/**
		 * @brief Check if it's a shallow ocean, regardless of temperature
		 * @param val The biome id to be checked against
		 * @return True if it's a shallow ocean
		*/
		inline static bool isShallowOcean(Sample val) noexcept {
			return val == STPBiomeRegistry::OCEAN.getID() || val == STPBiomeRegistry::FROZEN_OCEAN.getID()
				|| val == STPBiomeRegistry::WARM_OCEAN.getID() || val == STPBiomeRegistry::LUKEWARM_OCEAN.getID()
				|| val == STPBiomeRegistry::COLD_OCEAN.getID();
		}

		/**
		 * @brief Check if it's an ocean, regardless of biome variations
		 * @param val The biome id to be checked against
		 * @return True if it's an ocean biome.
		*/
		inline static bool isOcean(Sample val) noexcept {
			return STPBiomeRegistry::isShallowOcean(val) || val == STPBiomeRegistry::DEEP_OCEAN.getID()
				|| val == STPBiomeRegistry::DEEP_WARM_OCEAN.getID() || val == STPBiomeRegistry::DEEP_LUKEWARM_OCEAN.getID()
				|| val == STPBiomeRegistry::DEEP_COLD_OCEAN.getID() || val == STPBiomeRegistry::DEEP_FROZEN_OCEAN.getID();
		}

		/**
		 * @brief Check if it's a river biome, regardless of biome variations
		 * @param val The biome id to be checked against
		 * @return True if it's a river biome
		*/
		inline static bool isRiver(Sample val) noexcept {
			return val == STPBiomeRegistry::RIVER.getID() || val == STPBiomeRegistry::FROZEN_RIVER.getID();
		}

		/**
		 * @brief Get the precipitation type for this sample biome
		 * @param val The biome id
		 * @return The precipitation type of this biome
		*/
		static STPPrecipitationType getPrecipitationType(Sample val) {
			const STPBiome* const & biome = STPBiomeRegistry::REGISTRY[val];

			//we check for precipitation first, some biome like taiga, even it's cold but it's dry so it won't snow nor rain
			//of course we could have a more elegant model to determine the prec type, but let's just keep it simple
			if (biome->getPrecipitation() < 1.0f) {
				//desert and savana usually has prec less than 1.0
				return STPPrecipitationType::NONE;
			}

			if (biome->getTemperature() < 1.0f) {
				//snowy biome has temp less than 1.0
				return STPPrecipitationType::SNOW;
			}

			return STPPrecipitationType::RAIN;
		}
		
		/**
		 * @brief Apply the checker function to each samples
		 * @param function A function that takes in a sample variable and output bool for result
		 * @param samples... All checking samples
		 * @return True if all samples pass the checker function
		*/
		template <typename... S>
		static bool applyAll(bool (*checker)(Sample), const S... samples) {
			//type check
			static_assert(std::conjunction<std::is_same<Sample, S>...>::value, "Only sample values are allowed to be applied");

			const unsigned int len = sizeof...(S);
			if (len == 0) {
				return true;
			}

			//the power of functional programming
			//everyone must be true to be considered as true
			//c++17 fold function
			return ((*checker)(samples) && ...);
		}

		/**
		 * @brief Compare and swap operation
		 * @param comparator The value to be compared
		 * @param comparable The comparing value
		 * @param fallback If comparator is not equal to comparable, return this value
		 * @return Comparable if comparator equals comparable otherwise fallback value
		*/
		inline static Sample CAS(Sample comparator, Sample comparable, Sample fallback) noexcept {
			return comparator == comparable ? comparable : fallback;
		}
	};
	
}
#endif//_STP_BIOME_REGISTRY_H_