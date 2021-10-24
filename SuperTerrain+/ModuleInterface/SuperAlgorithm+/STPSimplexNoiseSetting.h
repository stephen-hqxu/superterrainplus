#pragma once
#ifndef _STP_SIMPLEX_NOISE_SETTING_H_
#define _STP_SIMPLEX_NOISE_SETTING_H_

#include <SuperAlgorithm+/STPAlgorithmDefine.h>
#include <SuperTerrain+/Environment/STPSetting.hpp>
//CUDA vector
#include <vector_functions.h>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPSimplexNoiseSettings specifies the simplex noise generator parameter for the simplex noise functions
	*/
	struct STPALGORITHMPLUS_HOST_API STPSimplexNoiseSetting : public STPSetting {
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
#endif//_STP_SIMPLEX_NOISE_SETTING_H_
