#pragma once
#ifndef _STP_SUN_H_
#define _STP_SUN_H_

#include <SuperRealism+/STPRealismDefine.h>
//Setting
#include "./Environment/STPSunSetting.h"
#include "./Environment/STPAtomsphereSetting.h"
//Rendering Engine
#include "./Object/STPProgramManager.h"

//GLM
#include <glm/vec3.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPSun is the main light source on the procedural terrain.
	 * It manages the position of the sun based on the time, rotates around the sky.
	 * It also allows, optionally, day-night cycle and switches light intensity.
	*/
	class STP_REALISM_API STPSun {
	public:

		/**
		 * @brief STPSunDirection contains angular information about a sun direction, all in radians.
		*/
		struct STPSunDirection {
		public:

			//The elevation angle is the angle between the sun and the horizontal. 
			//The elevation angle is similar to the zenith angle but it is measured from the horizontal rather than from the vertical, 
			//thus making the elevation angle = 90� - zenith.
			double Elevation;
			//The azimuth angle is the compass direction from which the sunlight is coming. 
			//At solar noon, the sun is always directly south in the northern hemisphere and directly north in the southern hemisphere.
			//At the equinoxes, the sun rises directly east and sets directly west regardless of the latitude, 
			//thus making the azimuth angles 90� at sunrise and 270� at sunset. 
			//In general however, the azimuth angle varies with the latitude and time of year.
			double Azimuth;

			//A unit vector of sun direction, this is calculate directly from elevation and azimuth angle.
			glm::dvec3 Direction;

		};

	private:

		const STPEnvironment::STPSunSetting& SunSetting;
		const STPEnvironment::STPAtomsphereSetting& SkySetting;

		//The time according to the position of the sun in the sky relative to one specific location on the ground, in tick
		size_t LocalSolarTime;
		//The number of day elapsed
		unsigned int Day;

		//The angle changed per tick, in radians
		const double AnglePerTick;
		//Denotes the tick at noon time, equals to day length by 2
		const size_t NoonTime;

	public:

		/**
		 * @brief Init the sun with settings.
		 * @param sun_setting The sun setting.
		 * @param sky_setting The sky setting.
		*/
		STPSun(const STPEnvironment::STPSunSetting&, const STPEnvironment::STPAtomsphereSetting&);

		STPSun(const STPSun&) = delete;

		STPSun(STPSun&&) = delete;

		STPSun& operator=(const STPSun&) = delete;

		STPSun& operator=(STPSun&&) = delete;

		~STPSun() = default;

		/**
		 * @brief Calculate the current direction of the sun.
		 * @return Information about the current sun direction.
		*/
		STPSunDirection currentDirection() const;

		/**
		 * @brief Bring the timer forward by a delta amount.
		 * @param delta The amount of tick to be added to the current LST.
		*/
		void deltaTick(size_t);

		/**
		 * @brief Get the current status of the sun.
		 * @param elevation The angle between the sun and the horizon.
		 * The elevation must be in radians.
		 * @return A value between 1.0 and -1.0.
		 * 1.0 -> sun is completely above the horizon.
		 * 0.0 -> horizon cuts the sun in half.
		 * -1.0 -> sun is completely below the horizon.
		*/
		double status(double) const;

	};

}
#endif//_STP_SUN_H_