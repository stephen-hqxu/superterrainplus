#pragma once
#ifndef _STP_SETTING_HPP_
#define _STP_SETTING_HPP_

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPEnvironment contains all configurations for each generators, like heightmap, normalmap, biomes, texture, etc.
	*/
	namespace STPEnvironment {

		/**
		 * @brief A base class for each Super Terrain + settings
		*/
		struct STPSetting {
		protected:

			/**
			 * @brief Init settings
			*/
			STPSetting() = default;

			~STPSetting() = default;

		public:

			/**
			 * @brief Validate each setting values and check if all settings are legal
			 * @return True if all settings are legal.
			*/
			virtual bool validate() const = 0;
		};

	}

}
#endif//_STP_SETTING_HPP_