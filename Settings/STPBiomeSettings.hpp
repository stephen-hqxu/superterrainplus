#pragma once
#ifndef _STP_BIOME_SETTINGS_HPP_
#define _STP_BIOME_SETTINGS_HPP_

#include "../World/Biome/STPBiome_def.h"
#include "STPSettings.hpp"
//String
#include <string>
using std::string;

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPSettings contains all configurations for each generators, like heightmap, normalmap, biomes, texture, etc.
	*/
	namespace STPSettings {

		/**
		 * @brief STPBiomeSettings stores settings for biome generation and texturing
		*/
		struct STPBiomeSettings : public STPSettings {
		public:

			//Identification and texture control
			//The id of this biome, for convention id equals the index of the biome registry, but it's free to choose the value of the id
			STPBiome::Sample ID;
			//The name of this biome
			string Name;
			//The temperature of this biome
			float Temperature;
			//The amount of rainfall in this biome
			float Precipitation;

			//Generation control
			//The base height of the biome
			float Depth;
			//The variation from the base height of the biome
			float Variation;

			/**
			 * @brief Init STPBiomeSettings with default values
			*/
			STPBiomeSettings() : STPSettings() {
				this->ID = 0;
				this->Name = "";
				this->Temperature = 0.0f;
				this->Precipitation = 0.0f;
				this->Depth = 0.0f;
				this->Variation = 0.0f;
			}

			~STPBiomeSettings() = default;

			bool validate() override {
				return this->Temperature >= 0.0f
					&& this->Precipitation >= 0.0f
					&& this->Depth >= 0.0f
					&& this->Variation >= 0.0f;
			}

		};

	}
}
#endif//_STP_BIOME_SETTINGS_HPP_