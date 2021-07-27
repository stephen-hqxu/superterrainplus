#pragma once
#ifndef _STP_BIOME_H_
#define _STP_BIOME_H_

//Biome define
#include <World/Diversity/STPBiomeDefine.h>
//Settings
#include "STPBiomeSettings.hpp"

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Sample;

	/**
	 * @brief STPBiome provides an abstract base class for each biome definition, including generation settings (altitude, variations) and
	 * texturing (based on climate, etc.).
	*/
	class STPBiome {
	private:

		//Settings for this biome
		STPBiomeSettings BiomeSettings;

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
		STPBiome(const STPBiomeSettings&);

		~STPBiome();

		/**
			* @brief Update the biome settings with a new one, if there is current no settings, it will be copied and stored
			* @param props The updating properties
		*/
		void updateProperties(const STPBiomeSettings&);

		/**
		 * @brief Get the biome properties stored.
		 * @return The biome properties
		*/
		const STPBiomeSettings& getProperties() const;

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
#endif//_STP_BIOME_H_