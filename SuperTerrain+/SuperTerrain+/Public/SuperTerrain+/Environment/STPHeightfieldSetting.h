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

		//Heightfield Generator Parameters
		//the seed used for any random operation during generation
		STPSeed_t Seed;

		//Hydraulic erosion parameters.
		STPRainDropSetting Erosion;

		void validate() const;
	};

}
#endif//_STP_HEIGHTFIELD_SETTING_H_