#pragma once
#ifndef _STP_BIOME_H_
#define _STP_BIOME_H_

//Biome define
#include "STPBiomeDefine.h"
//Settings
#include "../../Settings/STPBiomeSettings.hpp"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {
	/**
	 * @brief STPDiversity is a series of biome generation algorithm that allows user to define their own implementations
	*/
	namespace STPDiversity {
		/**
		 * @brief STPBiome provides an abstract base class for each biome definition, including generation settings (altitude, variations) and
		 * texturing (based on climate, etc.).
		*/
		class STPBiome {
		private:

			//Settings for this biome
			STPSettings::STPBiomeSettings BiomeSettings;

			//TODO: Textures for this biome

		public:

			/**
			 * @brief Init a new biome
			*/
			STPBiome();

			/**
			 * @brief Init a new biome with specified properties
			 * @param props Pointer to properties for this biome, it will be copied
			*/
			STPBiome(const STPSettings::STPBiomeSettings&);

			~STPBiome();

			/**
			 * @brief Update the biome settings with a new one, if there is current no settings, it will be copied and stored
			 * @param props The updating properties
			*/
			void updateProperties(const STPSettings::STPBiomeSettings&);

			/**
			 * @brief Get the id of this biome
			 * @return The biome id
			*/
			Sample getID() const;

			/**
			 * @brief Get the name of this biome
			 * @return The biome name
			*/
			std::string getName() const;

			/**
			 * @brief Get the temperature of this biome
			 * @return The temp
			*/
			float getTemperature() const;

			/**
			 * @brief Get the precipitation of this biome
			 * @return The precipitation
			*/
			float getPrecipitation() const;

		};
	}
}
#endif//_STP_BIOME_H_