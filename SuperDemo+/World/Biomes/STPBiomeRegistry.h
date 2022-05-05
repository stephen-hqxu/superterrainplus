#pragma once
#ifndef _STP_BIOME_REGISTRY_H_
#define _STP_BIOME_REGISTRY_H_

//ADT
#include <map>
#include <type_traits>
//Biome
#include "STPBiome.hpp"

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPBiomeRegistry contains all registered biome. Each biome is assigned to an unique sampling id, which uniquely identify different biome
	*/
	namespace STPBiomeRegistry {

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
		inline std::map<Sample, const STPBiome*> Registry;

		//A table of registered biome id, note that biomes are unordered

		/**
		 * @brief Ocean biome, base height is very low, with water filled atop
		*/
		inline STPBiome Ocean;
		/**
		 * @brief Deep ocean biome, similar to ocean biome but with even lower base height
		*/
		inline STPBiome DeepOcean;
		/**
		 * @brief Warm ocean biome, ocean located in hot biome
		*/
		inline STPBiome WarmOcean;
		/**
		 * @brief Lukewarm ocean biome, ocean located in moderate biome
		*/
		inline STPBiome LukewarmOcean;
		/**
		 * @brief Cold ocean biome, ocean located in cold biome
		*/
		inline STPBiome ColdOcean;
		/**
		 * @brief Frozen ocean biome, similar to ocean but water gets frozen in cold biomes or near cold biomes.
		*/
		inline STPBiome FrozenOcean;
		/**
		 * @brief Deep warm ocean biome, ocean located in hot biome with decreased base height
		*/
		inline STPBiome DeepWarmOcean;
		/**
		 * @brief Deep lukewarm ocean biome, ocean located in moderate biome with decreased base height
		*/
		inline STPBiome DeepLukewarmOcean;
		/**
		 * @brief Deep cold ocean biome, ocean located in cold biome with decreased base height
		*/
		inline STPBiome DeepColdOcean;
		/**
		 * @brief Deep frozen ocean biome, ocean located in super cold biome with decreased base height
		*/
		inline STPBiome DeepFrozenOcean;
		/**
		 * @brief Plains biome, a flat ground with little height variation, moderate temp and humidity
		*/
		inline STPBiome Plains;
		/**
		 * @brief Desert biome, everything is sand, it's super hot and dry AF
		*/
		inline STPBiome Desert;
		/**
		 * @brief Desert hills biome, located inside the desert with higher variation
		*/
		inline STPBiome DesertHills;
		/**
		 * @brief Mountain biome, base height is higher than most other biomes, with huge variation
		*/
		inline STPBiome Mountain;
		/**
		 * @brief Wooded mountain biome, located inside mountain but the ground is greener, elevation is pretty much the same
		*/
		inline STPBiome WoodedMountain;
		/**
		 * @brief Snowy mountain biome, located inside snowy tundra but with higher variation
		*/
		inline STPBiome SnowyMountain;
		/**
		 * @brief Forest biome, trees everywhere, warm and humid
		*/
		inline STPBiome Forest;
		/**
		 * @brief Forest hills biome, located inside forest, but with higher variation
		*/
		inline STPBiome ForestHills;
		/**
		 * @brief Taiga biome, mostly resemble a plain, but it's cold and wet.
		*/
		inline STPBiome Taiga;
		/**
		 * @brief Taiga gills biome, located inside taiga, but with higher variation
		*/
		inline STPBiome TaigaHills;
		/**
		 * @brief Snowy taiga biome, similar to taiga but it's colder
		*/
		inline STPBiome SnowyTaiga;
		/**
		 * @brief Snowy tundra biome, like taiga, but it's much colder and dryer, with less vegetations.
		*/
		inline STPBiome SnowyTundra;
		/**
		 * @brief River biome, one of the most special biome, it goes across the map randomly, and needs to be generated with separate algorithm
		*/
		inline STPBiome River;
		/**
		 * @brief Frozen river biome, similar to river but water gets frozen in cold biomes.
		*/
		inline STPBiome FrozenRiver;
		/**
		 * @brief Beach biome, one of the edge biome system, it connects various biomes with ocean biome.
		*/
		inline STPBiome Beach;
		/**
		 * @brief Snowy beach biome, similar to beach biome that acts as a connection between ocean and other biomes, but it apperas in cold and snowy area.
		*/
		inline STPBiome SnowyBeach;
		/**
		 * @brief Stone shore biome, similar to beach it can be found near the ocean, but it connects with mountain only
		*/
		inline STPBiome StoneShore;
		/**
		 * @brief Jungle biome, there are a lot of trees. It's super hot and wet
		*/
		inline STPBiome Jungle;
		/**
		 * @brief Jungle hills biome, located inside jungle, but with higher variation
		*/
		inline STPBiome JungleHills;
		/**
		 * @brief Savannah biome, basically like a plain, but it's hot and dry.
		*/
		inline STPBiome Savannah;
		/**
		 * @brief Savannah plateau biome, located inside savannah, but with higher variation
		*/
		inline STPBiome SavannahPlateau;
		/**
		 * @brief Swamp biome, it's usually hot and very dry, with low base height so there is a lot of water filling up, mostly found inside or near jungle.
		*/
		inline STPBiome Swamp;
		/**
		 * @brief Swamp hills biome, located inside swamp, but with higher variation
		*/
		inline STPBiome SwampHills;
		/**
		 * @brief Badlands biome, a biome that is super dry and full of harden clay and rock, and eroded
		*/
		inline STPBiome Badlands;
		/**
		 * @brief Badlands plateau biome, similar to badlands, but with higher variation
		*/
		inline STPBiome BadlandsPlateau;

		//Definitions of some biome utility functions

		/**
		 * @brief Call this function to register all biomes and fill up the biome registry
		*/
		inline void registerBiomes() {
			static bool initialised = false;
			if (initialised) {
				//do not re-initialise those biomes
				return;
			}

			//add all biomes to registry
			static auto reg_insert = [](const STPBiome& biome) -> void {
				STPBiomeRegistry::Registry.emplace(biome.ID, &biome);
				return;
			};
			//Oceans
			reg_insert(STPBiomeRegistry::Ocean);
			reg_insert(STPBiomeRegistry::DeepOcean);
			reg_insert(STPBiomeRegistry::WarmOcean);
			reg_insert(STPBiomeRegistry::LukewarmOcean);
			reg_insert(STPBiomeRegistry::ColdOcean);
			reg_insert(STPBiomeRegistry::FrozenOcean);
			reg_insert(STPBiomeRegistry::DeepWarmOcean);
			reg_insert(STPBiomeRegistry::DeepLukewarmOcean);
			reg_insert(STPBiomeRegistry::DeepColdOcean);
			reg_insert(STPBiomeRegistry::DeepFrozenOcean);
			//Rivers
			reg_insert(STPBiomeRegistry::River);
			reg_insert(STPBiomeRegistry::FrozenRiver);
			//Lands
			reg_insert(STPBiomeRegistry::Plains);
			reg_insert(STPBiomeRegistry::Desert);
			reg_insert(STPBiomeRegistry::Mountain);
			reg_insert(STPBiomeRegistry::Forest);
			reg_insert(STPBiomeRegistry::Taiga);
			reg_insert(STPBiomeRegistry::SnowyTaiga);
			reg_insert(STPBiomeRegistry::SnowyTundra);
			reg_insert(STPBiomeRegistry::Jungle);
			reg_insert(STPBiomeRegistry::Savannah);
			reg_insert(STPBiomeRegistry::Swamp);
			reg_insert(STPBiomeRegistry::Badlands);
			//Hills
			reg_insert(STPBiomeRegistry::DesertHills);
			reg_insert(STPBiomeRegistry::TaigaHills);
			reg_insert(STPBiomeRegistry::WoodedMountain);
			reg_insert(STPBiomeRegistry::SnowyMountain);
			reg_insert(STPBiomeRegistry::ForestHills);
			reg_insert(STPBiomeRegistry::JungleHills);
			reg_insert(STPBiomeRegistry::SavannahPlateau);
			reg_insert(STPBiomeRegistry::SwampHills);
			reg_insert(STPBiomeRegistry::BadlandsPlateau);
			//Edges and Shores
			reg_insert(STPBiomeRegistry::Beach);
			reg_insert(STPBiomeRegistry::SnowyBeach);
			reg_insert(STPBiomeRegistry::StoneShore);

			initialised = true;
		}

		/**
		 * @brief Check if it's a shallow ocean, regardless of temperature
		 * @param val The biome id to be checked against
		 * @return True if it's a shallow ocean
		*/
		inline bool isShallowOcean(Sample val) noexcept {
			return val == STPBiomeRegistry::Ocean.ID || val == STPBiomeRegistry::FrozenOcean.ID
				|| val == STPBiomeRegistry::WarmOcean.ID || val == STPBiomeRegistry::LukewarmOcean.ID
				|| val == STPBiomeRegistry::ColdOcean.ID;
		}

		/**
		 * @brief Check if it's an ocean, regardless of biome variations
		 * @param val The biome id to be checked against
		 * @return True if it's an ocean biome.
		*/
		inline bool isOcean(Sample val) noexcept {
			return STPBiomeRegistry::isShallowOcean(val) || val == STPBiomeRegistry::DeepOcean.ID
				|| val == STPBiomeRegistry::DeepWarmOcean.ID || val == STPBiomeRegistry::DeepLukewarmOcean.ID
				|| val == STPBiomeRegistry::DeepColdOcean.ID || val == STPBiomeRegistry::DeepFrozenOcean.ID;
		}

		/**
		 * @brief Check if it's a river biome, regardless of biome variations
		 * @param val The biome id to be checked against
		 * @return True if it's a river biome
		*/
		inline bool isRiver(Sample val) noexcept {
			return val == STPBiomeRegistry::River.ID || val == STPBiomeRegistry::FrozenRiver.ID;
		}

		/**
		 * @brief Get the precipitation type for this sample biome
		 * @param val The biome id
		 * @return The precipitation type of this biome
		*/
		inline STPPrecipitationType getPrecipitationType(Sample val) {
			const STPBiome* const& biome = STPBiomeRegistry::Registry[val];

			//we check for precipitation first, some biome like taiga, even it's cold but it's dry so it won't snow nor rain
			//of course we could have a more elegant model to determine the precipitation type, but let's just keep it simple
			if (biome->Precipitation < 1.0f) {
				//desert and savannah usually has precipitation less than 1.0
				return STPPrecipitationType::NONE;
			}

			if (biome->Temperature < 1.0f) {
				//snowy biome has temp less than 1.0
				return STPPrecipitationType::SNOW;
			}

			return STPPrecipitationType::RAIN;
		}

		/**
		 * @brief Apply the checker function to each samples
		 * @param function A function that takes in a sample variable and output boolean for result
		 * @param samples... All checking samples
		 * @return True if all samples pass the checker function
		*/
		template <typename... S>
		inline bool applyAll(bool (*checker)(Sample), const S... samples) {
			//type check
			static_assert(std::conjunction<std::is_same<Sample, S>...>::value, "Only sample values are allowed to be applied");

			if constexpr (sizeof...(S) == 0) {
				return true;
			}

			return ((*checker)(samples) && ...);
		}

		/**
		 * @brief Compare and swap operation
		 * @param comparator The value to be compared
		 * @param comparable The comparing value
		 * @param fallback If comparator is not equal to comparable, return this value
		 * @return Comparable if comparator equals comparable otherwise fallback value
		*/
		inline Sample CAS(Sample comparator, Sample comparable, Sample fallback) noexcept {
			return comparator == comparable ? comparable : fallback;
		}
	};

}
#endif//_STP_BIOME_REGISTRY_H_