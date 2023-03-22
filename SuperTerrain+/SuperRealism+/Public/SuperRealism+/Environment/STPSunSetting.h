#pragma once
#ifndef _STP_SUN_SETTING_H_
#define _STP_SUN_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPSunSetting stores all settings for STPSun, the main light source of the world.
	*/
	struct STP_REALISM_API STPSunSetting {
	public:

		//The length of a day in second.
		unsigned int DayLength;
		//The second in a day set to at the beginning.
		unsigned int DayStart;
		//The number of day per year.
		unsigned int YearLength;
		//The day in a year set to at the beginning.
		unsigned int YearStart;

		//The angle between an object's rotational axis and its orbital axis, in radians, which is the line perpendicular to its orbital plane; 
		//equivalently, it is the angle between its equatorial plane and orbital plane.
		double Obliquity;
		//A geographic coordinate that specifies the north–south position of a point on the planet surface.
		//Latitude is an angle which ranges from 0 at the Equator to 90 (North or South) at the poles.
		//Remember to convert it into radians.
		double Latitude;

		void validate() const;

	};

}
#endif//_STP_SUN_SETTING_H_