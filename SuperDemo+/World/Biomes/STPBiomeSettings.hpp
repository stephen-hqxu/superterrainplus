#pragma once
#ifndef _STP_BIOME_SETTINGS_HPP_
#define _STP_BIOME_SETTINGS_HPP_

#include "STPBiomeProperty.hpp"
#include <SuperTerrain+/Environment/STPSetting.hpp>
#include <SuperTerrain+/World/Diversity/STPBiomeDefine.h>
//String
#include <string>

namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPBiomeSettings stores all biome settings, including information about the biome and generation parameters
	*/
	struct STPBiomeSettings : public STPBiomeProperty, public SuperTerrainPlus::STPEnvironment::STPSetting {
	public:

		//Identification and texture control
		//The id of this biome, for convention id equals the index of the biome registry, but it's free to choose the value of the id
		SuperTerrainPlus::STPDiversity::Sample ID;
		//The name of this biome
		std::string Name;
		//The temperature of this biome
		float Temperature;
		//The amount of rainfall in this biome
		float Precipitation;

		/**
			* @brief Init STPBiomeSettings with default values
		*/
		STPBiomeSettings() : STPSetting() {
			this->ID = 0;
			this->Name = "";
			this->Temperature = 0.0f;
			this->Precipitation = 0.0f;
			this->Depth = 0.0f;
			this->Variation = 0.0f;
		}

		~STPBiomeSettings() = default;

		bool validate() const override {
			return this->Temperature >= 0.0f
				&& this->Precipitation >= 0.0f
				&& this->Depth >= 0.0f
				&& this->Variation >= 0.0f;
		}

	};
}
#endif//_STP_BIOME_SETTINGS_HPP_