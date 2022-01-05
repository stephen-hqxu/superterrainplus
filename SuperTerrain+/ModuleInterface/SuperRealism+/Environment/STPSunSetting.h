#pragma once
#ifndef _STP_SUN_SETTING_H_
#define _STP_SUN_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>
//Base Environment
#include <SuperTerrain+/Environment/STPSetting.hpp>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPSunSetting stores all settings for STPSun, the main light source of the world.
	*/
	struct STP_REALISM_API STPSunSetting : public STPSetting {
	public:

		//The number of tick per day
		//Tick is the unit of time, however the actual meaning of tick is implementation defined (by the user).
		unsigned long long DayLength;
		//The tick set to at the start of the sun application
		unsigned long long DayStartOffset;
		//The number of day per year
		unsigned int YearLength;

		//The angle between an object's rotational axis and its orbital axis, in radians, which is the line perpendicular to its orbital plane; 
		//equivalently, it is the angle between its equatorial plane and orbital plane.
		double Obliquity;
		//A geographic coordinate that specifies the north–south position of a point on the planet surface.
		//Latitude is an angle which ranges from 0 at the Equator to 90 (North or South) at the poles.
		//Remember to convert it into radians.
		double Latitude;

		/**
		 * @brief Init STPSunSetting with default settings.
		*/
		STPSunSetting();

		~STPSunSetting() = default;

		bool validate() const override;

	};

}
#endif//_STP_SUN_SETTING_H_