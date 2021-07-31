#pragma once
#ifndef _STP_SIMPLEX_NOISE_SETTING_H_
#define _STP_SIMPLEX_NOISE_SETTING_H_

#include "STPAlgorithmDefine.h"
#include <Environment/STPSetting.hpp>
//CUDA vector
#include <vector_functions.h>

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
		 * @brief STPSimplexNoiseSettings specifies the simplex noise generator parameter for the simplex noise functions
		*/
		struct STPALGORITHMPLUS_HOST_API STPSimplexNoiseSetting: public STPSetting {
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
			STPSimplexNoiseSetting();

			~STPSimplexNoiseSetting() = default;

			bool validate() const override;

		};
	}
}
#endif//_STP_SIMPLEX_NOISE_SETTING_H_
