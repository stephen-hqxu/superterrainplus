#pragma once
#ifndef _STP_SEED_MIXER_H_
#define _STP_SEED_MIXER_H_

#include <STPCoreDefine.h>
//Biome define
#include "STPBiomeDefine.h"

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
		 * @brief STPSeedMixer mixes two seeds together to form a new seed
		*/
		class STP_API STPSeedMixer final {
		private:

			//No initialisation for a class with static functions only
			STPSeedMixer();

			~STPSeedMixer();

		public:

			/**
			 * @brief Mix seed with a factor to achieve a degree of randomness to form a new seed. This function guaratees that if two same values are the same,
			 * the returning seed will always be the same
			 * @param s The seed
			 * @param fac The factor that is used to mix
			 * @return The mixed seed
			*/
			static Seed mixSeed(Seed, long long);

		};
	}
}
#endif//_STP_SEED_MIXER_H_