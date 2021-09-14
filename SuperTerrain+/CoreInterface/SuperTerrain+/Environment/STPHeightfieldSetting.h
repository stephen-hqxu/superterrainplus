#pragma once
#ifndef _STP_HEIGHTFIELD_SETTING_H_
#define _STP_HEIGHTFIELD_SETTING_H_

#include "STPRainDropSetting.cuh"

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
		 * @brief STPHeightfieldSettings stores all heightfield parameters for compute launch
		*/
		struct STP_API STPHeightfieldSetting: public STPRainDropSetting {
		public:

			//Heightfield Generator Parameters
			//the seed used for any random opertaion during generation
			unsigned long long Seed;
			//Normalmap Parameters
			//Control the strength of z component of the normal map, the greater, the more the normal pointing towards the surface
			float Strength;

			//Hydraulic Erosion Parameters are inherited from super class

			//STPRainDropSetting is non-copiable

			/**
			 * @brief Init STPHeightfieldSetting with defaults
			*/
			STPHeightfieldSetting();

			STPHeightfieldSetting(const STPHeightfieldSetting&) = delete;

			STPHeightfieldSetting(STPHeightfieldSetting&&) noexcept = default;

			STPHeightfieldSetting& operator=(const STPHeightfieldSetting&) = delete;

			STPHeightfieldSetting& operator=(STPHeightfieldSetting&&) noexcept = default;

			~STPHeightfieldSetting() = default;

			bool validate() const override;
		};

	}

}
#endif//_STP_HEIGHTFIELD_SETTING_H_