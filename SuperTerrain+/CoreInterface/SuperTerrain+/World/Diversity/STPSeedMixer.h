#pragma once
#ifndef _STP_SEED_MIXER_H_
#define _STP_SEED_MIXER_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Biome define
#include "STPBiomeDefine.h"

namespace SuperTerrainPlus::STPDiversity {

	/**
	 * @brief STPSeedMixer mixes two seeds together to form a new seed
	*/
	namespace STPSeedMixer {

		/**
		 * @brief Mix seed with a factor to achieve a degree of randomness to form a new seed. This function guaratees that if two same values are the same,
		 * the returning seed will always be the same
		 * @param s The seed
		 * @param fac The factor that is used to mix
		 * @return The mixed seed
		*/
		STP_API Seed mixSeed(Seed, long long);

	};
}
#endif//_STP_SEED_MIXER_H_