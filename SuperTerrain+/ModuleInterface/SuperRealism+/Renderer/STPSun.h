#pragma once
#ifndef _STP_SUN_H_
#define _STP_SUN_H_

#include <SuperRealism+/STPRealismDefine.h>
//Setting
#include "../Environment/STPSunSetting.h"
#include "../Environment/STPAtmosphereSetting.h"
//Rendering Engine
#include "../Object/STPProgramManager.h"
#include "../Object/STPBuffer.h"
#include "../Object/STPVertexArray.h"
#include "../Utility/STPLogStorage.hpp"

//GLM
#include <glm/vec3.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPSun is the main light source on the procedural terrain.
	 * It manages the position of the sun based on the time, rotates around the sky.
	 * It also allows, optionally, day-night cycle and switches light intensity.
	 * Atmoshperic scattering produced by the sun is also simulated by rendering the sun as an environmental light source.
	*/
	class STP_REALISM_API STPSun {
	private:

		const STPEnvironment::STPSunSetting& SunSetting;

		//Those buffers specify the ray direction from the camera
		STPBuffer RayDirectionBuffer, RayDirectionIndex, SkyRenderCommand;
		STPVertexArray RayDirectionArray;
		//Shaders
		STPProgramManager SkyRenderer;

		//The number of day elapsed
		//The integer part is the number of day, the fractional part is the local solar time.
		//The time according to the position of the sun in the sky relative to one specific location on the ground, in tick
		double Day;

		//The angle changed per tick, in radians
		const double AnglePerTick;
		//Denotes the tick at noon time, equals to day length by 2
		const unsigned long long NoonTime;

		//Records the most recent calculation to avoid recomputation.
		//The elevation angle is the angle between the sun and the horizontal. 
		//The elevation angle is similar to the zenith angle but it is measured from the horizontal rather than from the vertical, 
		//thus making the elevation angle = 90° - zenith.
		//The azimuth angle is the compass direction from which the sunlight is coming. 
		//At solar noon, the sun is always directly south in the northern hemisphere and directly north in the southern hemisphere.
		//At the equinoxes, the sun rises directly east and sets directly west regardless of the latitude, 
		//thus making the azimuth angles 90° at sunrise and 270° at sunset. 
		//In general however, the azimuth angle varies with the latitude and time of year.
		glm::vec3 SunDirectionCache;

	public:

		//The log for STPSun, coming from sun and sky renderer.
		typedef STPLogStorage<3ull> STPSunLog;

		/**
		 * @brief Init the sun with settings.
		 * @param sun_setting The sun setting.
		 * @param log Logs output from the shader compilation.
		*/
		STPSun(const STPEnvironment::STPSunSetting&, STPSunLog&);

		STPSun(const STPSun&) = delete;

		STPSun(STPSun&&) = delete;

		STPSun& operator=(const STPSun&) = delete;

		STPSun& operator=(STPSun&&) = delete;

		~STPSun() = default;

		/**
		 * @brief Get the current sun direction.
		 * @return The pointer to the current calulated sun direction cache, 
		 * A unit vector of sun direction, this is calculate directly from elevation and azimuth angle.
		*/
		const glm::vec3& sunDirection() const;

		/**
		 * @brief Bring the timer forward by a delta amount and update the sun position.
		 * @param tick The number of tick to be added to the current LST.
		*/
		void advanceTick(unsigned long long);

		/**
		 * @brief Get the current status of the sun.
		 * @param elevation The y component of the normalised sun direction.
		 * The elevation must be between -1.0f and 1.0f.
		 * @return A value between 1.0 and -1.0.
		 * 1.0 -> sun is completely above the horizon.
		 * 0.0 -> horizon cuts the sun in half.
		 * -1.0 -> sun is completely below the horizon.
		*/
		float status(float) const;

		/**
		 * @brief Update the sky renderer with new atmoshpere setting.
		 * @param sky_setting The sky setting. 
		*/
		void setAtmoshpere(const STPEnvironment::STPAtmosphereSetting&);

		/**
		 * @brief Render the sun with atmospheric scattering effect.
		*/
		void operator()() const;

	};

}
#endif//_STP_SUN_H_