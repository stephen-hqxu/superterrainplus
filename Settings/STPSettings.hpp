#pragma once
#ifndef _STP_SETTINGS_HPP_
#define _STP_SETTINGS_HPP_

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
		 * @brief A base class for each Super Terrain + settings
		*/
		struct STPSettings {
		protected:

			/**
			 * @brief Init settings
			*/
			STPSettings() = default;

			~STPSettings() = default;

		public:

			/**
			 * @brief Validate each setting values and check if all settings are legal
			 * @return True if all settings are legal.
			*/
			virtual bool validate() = 0;
		};

	}

}
#endif//_STP_SETTINGS_HPP_