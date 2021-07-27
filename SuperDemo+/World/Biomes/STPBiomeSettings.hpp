#pragma once
#ifndef _STP_BIOME_SETTINGS_HPP_
#define _STP_BIOME_SETTINGS_HPP_

#include <World/Diversity/STPBiomeDefine.h>
#include <Environment/STPSetting.hpp>
//String
#include <string>

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPBiomeSettings stores settings for biome generation and texturing
	*/
	struct STPBiomeSettings : public SuperTerrainPlus::STPEnvironment::STPSetting {
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

		//Generation control
		//Determine the zooming of the noise map
		float Scale;
		//Control how many heightmap will be conbined
		unsigned int Octave;
		//Control how the amplitude will be changed in each octave. Range (0,1)
		float Persistence;
		//Control how the frequency will be changed in each octave.
		float Lacunarity;
		//The base height of the biome
		float Depth;
		//The variation from the base height of the biome
		float Variation;

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