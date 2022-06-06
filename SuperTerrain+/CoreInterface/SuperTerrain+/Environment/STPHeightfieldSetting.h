#pragma once
#ifndef _STP_HEIGHTFIELD_SETTING_H_
#define _STP_HEIGHTFIELD_SETTING_H_

#include "STPRainDropSetting.h"

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPHeightfieldSettings stores all heightfield parameters for compute launch
	*/
	struct STP_API STPHeightfieldSetting : public STPRainDropSetting {
	public:

		//Heightfield Generator Parameters
		//the seed used for any random operation during generation
		unsigned long long Seed;

		//Hydraulic Erosion Parameters are inherited from super class

		/**
		 * @brief Init STPHeightfieldSetting with defaults
		*/
		STPHeightfieldSetting();

		~STPHeightfieldSetting() = default;

		bool validate() const override;
	};

}
#endif//_STP_HEIGHTFIELD_SETTING_H_