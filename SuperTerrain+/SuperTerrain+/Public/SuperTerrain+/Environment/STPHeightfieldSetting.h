#pragma once
#ifndef _STP_HEIGHTFIELD_SETTING_H_
#define _STP_HEIGHTFIELD_SETTING_H_

#include "../World/STPWorldMapPixelFormat.hpp"

#include "STPRainDropSetting.h"

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPHeightfieldSettings stores all heightfield parameters for compute launch
	*/
	struct STP_API STPHeightfieldSetting {
	public:

		//The seed used for randomly rolling the starting position of individual raindrop.
		STPSeed_t Seed;

		//Hydraulic erosion parameters.
		STPRainDropSetting Erosion;

		void validate() const;

	};

}
#endif//_STP_HEIGHTFIELD_SETTING_H_