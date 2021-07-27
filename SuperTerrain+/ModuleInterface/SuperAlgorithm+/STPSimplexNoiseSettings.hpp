#pragma once
#ifndef _STP_SIMPLEX_NOISE_SETTINGS_HPP_
#define _STP_SIMPLEX_NOISE_SETTINGS_HPP_

#include <Settings/STPSetting.hpp>
//CUDA vector
#include <vector_functions.h>

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
		 * @brief STPSimplexNoiseSettings specifies the simplex noise generator parameter for the simplex noise functions
		*/
		struct STPSimplexNoiseSettings: public STPSetting {
		public:

			//Determine the seed used for the RNG
			unsigned long long Seed;
			//Determine how many gradient stretch will have, default is 8, each of them will be 45 degree apart.
			//Higher value will make the terrain looks more random with less systematic pattern
			unsigned int Distribution;
			//Determine the offset of the angle for the gradient table, in degree
			//This will generally rotate the terrain
			double Offset;

			/**
			 * @brief Init the simplex noise settings with default values
			*/
			STPSimplexNoiseSettings() : STPSetting() {
				//Loading default value
				this->Seed = 0u;
				this->Distribution = 8u;
				this->Offset = 45.0;
			}

			~STPSimplexNoiseSettings() = default;

			bool validate() const override {
				return this->Distribution != 0
					&& this->Offset >= 0.0
					&& this->Offset < 360.0;
			}

		};
	}
}
#endif//_STP_SIMPLEX_NOISE_SETTINGS_HPP_
