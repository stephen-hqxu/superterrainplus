#pragma once
#ifndef _STP_STARFIELD_SETTING_H_
#define _STP_STARFIELD_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>
#include <SuperTerrain+/World/STPWorldMapPixelFormat.hpp>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPStarfieldSetting contains settings for rendering a procedural starfield.
	*/
	struct STP_REALISM_API STPStarfieldSetting {
	public:

		//A random seed to control the shining animation of stars.
		STPSeed_t Seed;

		//Control how likely star will appear at each position.
		//Higher likelihood makes the starfield denser.
		float InitialLikelihood, OctaveLikelihoodMultiplier;
		//Control the size of each star.
		float InitialScale, OctaveScaleMultiplier;
		//Each generated star has a soft edge fading off away from the star centre,
		//this parameter controls the softness of the edge.
		float EdgeDistanceFalloff;
		//Specifies the speed stars shine.
		float ShineSpeed;
		//An intensity multiplier to the output of the star colour.
		float LuminosityMultiplier;
		//Specifies a minimum direction elevation for which stars should be rendered.
		//Note that ray elevation is within the range [-1.0, 1.0].
		float MinimumAltitude;

		//Specifies the number of iteration for starfield generation.
		//More octave gives a larger number of star but also degrades the performance.
		unsigned int Octave;

		void validate() const;

	};
}
#endif//_STP_STARFIELD_SETTING_H_