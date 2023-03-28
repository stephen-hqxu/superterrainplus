#pragma once
#ifndef _STP_BIOME_REGISTRY_H_
#define _STP_BIOME_REGISTRY_H_

//ADT
#include <unordered_map>
//Biome
#include "STPBiome.hpp"

namespace STPDemo {

	/**
	 * @brief STPBiomeRegistry contains all registered biome. Each biome is assigned to an unique sampling id, which uniquely identify different biome
	*/
	namespace STPBiomeRegistry {

		using SuperTerrainPlus::STPSample_t;

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
		extern std::unordered_map<STPSample_t, const STPBiome*> Registry;

		//A table of registered biome id, note that biomes are unordered

		/**
		 * @brief Ocean biome, base height is very low, with water filled atop
		*/
		extern STPBiome Ocean;
		/**
		 * @brief Deep ocean biome, similar to ocean biome but with even lower base height
		*/
		extern STPBiome DeepOcean;
		/**
		 * @brief Warm ocean biome, ocean located in hot biome
		*/
		extern STPBiome WarmOcean;
		/**
		 * @brief Lukewarm ocean biome, ocean located in moderate biome
		*/
		extern STPBiome LukewarmOcean;
		/**
		 * @brief Cold ocean biome, ocean located in cold biome
		*/
		extern STPBiome ColdOcean;
		/**
		 * @brief Frozen ocean biome, similar to ocean but water gets frozen in cold biomes or near cold biomes.
		*/
		extern STPBiome FrozenOcean;
		/**
		 * @brief Deep warm ocean biome, ocean located in hot biome with decreased base height
		*/
		extern STPBiome DeepWarmOcean;
		/**
		 * @brief Deep lukewarm ocean biome, ocean located in moderate biome with decreased base height
		*/
		extern STPBiome DeepLukewarmOcean;
		/**
		 * @brief Deep cold ocean biome, ocean located in cold biome with decreased base height
		*/
		extern STPBiome DeepColdOcean;
		/**
		 * @brief Deep frozen ocean biome, ocean located in super cold biome with decreased base height
		*/
		extern STPBiome DeepFrozenOcean;
		/**
		 * @brief Plains biome, a flat ground with little height variation, moderate temp and humidity
		*/
		extern STPBiome Plains;
		/**
		 * @brief Desert biome, everything is sand, it's super hot and dry AF
		*/
		extern STPBiome Desert;
		/**
		 * @brief Desert hills biome, located inside the desert with higher variation
		*/
		extern STPBiome DesertHills;
		/**
		 * @brief Mountain biome, base height is higher than most other biomes, with huge variation
		*/
		extern STPBiome Mountain;
		/**
		 * @brief Wooded mountain biome, located inside mountain but the ground is greener, elevation is pretty much the same
		*/
		extern STPBiome WoodedMountain;
		/**
		 * @brief Snowy mountain biome, located inside snowy tundra but with higher variation
		*/
		extern STPBiome SnowyMountain;
		/**
		 * @brief Forest biome, trees everywhere, warm and humid
		*/
		extern STPBiome Forest;
		/**
		 * @brief Forest hills biome, located inside forest, but with higher variation
		*/
		extern STPBiome ForestHills;
		/**
		 * @brief Taiga biome, mostly resemble a plain, but it's cold and wet.
		*/
		extern STPBiome Taiga;
		/**
		 * @brief Taiga gills biome, located inside taiga, but with higher variation
		*/
		extern STPBiome TaigaHills;
		/**
		 * @brief Snowy taiga biome, similar to taiga but it's colder
		*/
		extern STPBiome SnowyTaiga;
		/**
		 * @brief Snowy tundra biome, like taiga, but it's much colder and dryer, with less vegetations.
		*/
		extern STPBiome SnowyTundra;
		/**
		 * @brief River biome, one of the most special biome, it goes across the map randomly, and needs to be generated with separate algorithm
		*/
		extern STPBiome River;
		/**
		 * @brief Frozen river biome, similar to river but water gets frozen in cold biomes.
		*/
		extern STPBiome FrozenRiver;
		/**
		 * @brief Beach biome, one of the edge biome system, it connects various biomes with ocean biome.
		*/
		extern STPBiome Beach;
		/**
		 * @brief Snowy beach biome, similar to beach biome that acts as a connection between ocean and other biomes, but it apperas in cold and snowy area.
		*/
		extern STPBiome SnowyBeach;
		/**
		 * @brief Stone shore biome, similar to beach it can be found near the ocean, but it connects with mountain only
		*/
		extern STPBiome StoneShore;
		/**
		 * @brief Jungle biome, there are a lot of trees. It's super hot and wet
		*/
		extern STPBiome Jungle;
		/**
		 * @brief Jungle hills biome, located inside jungle, but with higher variation
		*/
		extern STPBiome JungleHills;
		/**
		 * @brief Savannah biome, basically like a plain, but it's hot and dry.
		*/
		extern STPBiome Savannah;
		/**
		 * @brief Savannah plateau biome, located inside savannah, but with higher variation
		*/
		extern STPBiome SavannahPlateau;
		/**
		 * @brief Swamp biome, it's usually hot and very dry, with low base height so there is a lot of water filling up, mostly found inside or near jungle.
		*/
		extern STPBiome Swamp;
		/**
		 * @brief Swamp hills biome, located inside swamp, but with higher variation
		*/
		extern STPBiome SwampHills;
		/**
		 * @brief Badlands biome, a biome that is super dry and full of harden clay and rock, and eroded
		*/
		extern STPBiome Badlands;
		/**
		 * @brief Badlands plateau biome, similar to badlands, but with higher variation
		*/
		extern STPBiome BadlandsPlateau;

		//Definitions of some biome utility functions

		/**
		 * @brief Call this function to register all biomes and fill up the biome registry
		*/
		void registerBiomes();

		/**
		 * @brief Check if it's a shallow ocean, regardless of temperature
		 * @param val The biome id to be checked against
		 * @return True if it's a shallow ocean
		*/
		bool isShallowOcean(STPSample_t) noexcept;

		/**
		 * @brief Check if it's an ocean, regardless of biome variations
		 * @param val The biome id to be checked against
		 * @return True if it's an ocean biome.
		*/
		bool isOcean(STPSample_t) noexcept;

		/**
		 * @brief Check if it's a river biome, regardless of biome variations
		 * @param val The biome id to be checked against
		 * @return True if it's a river biome
		*/
		bool isRiver(STPSample_t) noexcept;

		/**
		 * @brief Get the precipitation type for this sample biome
		 * @param val The biome id
		 * @return The precipitation type of this biome
		*/
		STPPrecipitationType getPrecipitationType(STPSample_t);

		/**
		 * @brief Apply the checker function to each samples
		 * @param function A function that takes in a sample variable and output boolean for result
		 * @param samples... All checking samples
		 * @return True if all samples pass the checker function
		*/
		template <typename... S>
		inline bool applyAll(bool (*checker)(STPSample_t), const S... samples) noexcept {
			if constexpr (sizeof...(S) == 0) {
				return true;
			} else {
				return ((*checker)(samples) && ...);
			}
		}

		/**
		 * @brief Compare and swap operation
		 * @param comparator The value to be compared
		 * @param comparable The comparing value
		 * @param fallback If comparator is not equal to comparable, return this value
		 * @return Comparable if comparator equals comparable otherwise fallback value
		*/
		STPSample_t CAS(STPSample_t, STPSample_t, STPSample_t) noexcept;
	}

}
#endif//_STP_BIOME_REGISTRY_H_