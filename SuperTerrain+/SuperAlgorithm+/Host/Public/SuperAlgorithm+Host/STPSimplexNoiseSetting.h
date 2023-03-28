#pragma once
#ifndef _STP_SIMPLEX_NOISE_SETTING_H_
#define _STP_SIMPLEX_NOISE_SETTING_H_

#include <SuperAlgorithm+Host/STPAlgorithmDefine.h>
#include <SuperTerrain+/World/STPWorldMapPixelFormat.hpp>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPSimplexNoiseSettings specifies the simplex noise generator parameter for the simplex noise functions
	*/
	struct STP_ALGORITHM_HOST_API STPSimplexNoiseSetting {
	public:

		//Determine the seed used for the RNG
		STPSeed_t Seed;
		//Determine how many gradient stretch will have, default is 8, each of them will be 45 degree apart.
		//Higher value will make the terrain looks more random with less systematic pattern
		unsigned int Distribution;
		//Determine the offset of the angle for the gradient table, in radians
		//This will generally rotate the terrain
		double Offset;

		void validate() const;

	};
}
#endif//_STP_SIMPLEX_NOISE_SETTING_H_
