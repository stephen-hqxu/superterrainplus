#pragma once
#ifndef _STP_SUN_H_
#define _STP_SUN_H_

#include <SuperRealism+/STPRealismDefine.h>
//Setting
#include "../../Environment/STPSunSetting.h"
#include "../../Environment/STPAtmosphereSetting.h"
//Rendering Engine
#include "../../Object/STPProgramManager.h"
#include "../../Object/STPBuffer.h"
#include "../../Object/STPVertexArray.h"

#include "../Light/STPLightSpectrum.h"
#include "../STPSceneObject.h"

//GLM
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPSun is the main light source on the procedural terrain.
	 * It manages the position of the sun based on the time, rotates around the sky.
	 * It also allows, optionally, day-night cycle and switches light intensity.
	 * Atmospheric scattering produced by the sun is also simulated by rendering the sun as an environmental light source.
	*/
	class STP_REALISM_API STPSun : public STPSceneObject::STPEnvironmentObject {
	public:

		//A pair of two equivalent types.
		template<class T>
		using STPBundledData = std::pair<T, T>;

	private:

		const STPEnvironment::STPSunSetting SunSetting;

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
		//Records the most recent calculation to avoid re-computation.
		//The elevation angle is the angle between the sun and the horizontal. 
		//The elevation angle is similar to the zenith angle but it is measured from the horizontal rather than from the vertical, 
		//thus making the elevation angle = 90° - zenith.
		//The azimuth angle is the compass direction from which the sunlight is coming. 
		//At solar noon, the sun is always directly south in the northern hemisphere and directly north in the southern hemisphere.
		//At the equinoxes, the sun rises directly east and sets directly west regardless of the latitude, 
		//thus making the azimuth angles 90° at sunrise and 270° at sunset. 
		//In general however, the azimuth angle varies with the latitude and time of year.
		glm::vec3 SunDirectionCache;

		//A compute shader to generate an approximation of sky and sun spectrum
		mutable STPProgramManager SpectrumEmulator;
		//Record the elevation of sun direction domain of the spectrum.
		STPBundledData<float> SpectrumDomainElevation;

		STPOpenGL::STPint SunPositionLocation;

		/**
		 * @brief Send new atmosphere settings as uniforms to the destination program.
		 * @param program The program where the uniforms will be sent to.
		 * @param atmo_setting The atmosphere setting to be updated.
		*/
		static void updateAtmosphere(STPProgramManager&, const STPEnvironment::STPAtmosphereSetting&);

	public:

		/**
		 * @brief Init the sun with settings.
		 * @param sun_setting The sun setting. The setting will be copied to the underlying instance.
		 * @param spectrum_domain For sun spectrum emulation.
		 * Specifies The sun direction for the first iteration and the last iteration.
		 * Sun direction in between will be interpolated.
		*/
		STPSun(const STPEnvironment::STPSunSetting&, const STPBundledData<glm::vec3>&);

		STPSun(const STPSun&) = delete;

		STPSun(STPSun&&) = delete;

		STPSun& operator=(const STPSun&) = delete;

		STPSun& operator=(STPSun&&) = delete;

		~STPSun() = default;

		/**
		 * @brief Get the direction of the sun.
		 * @return The pointer to the sun direction, which is associated with the current sun instance.
		*/
		const glm::vec3& sunDirection() const;

		/**
		 * @brief Bring the timer forward by a delta amount and update the sun position.
		 * @param tick The number of tick to be added to the current LST.
		*/
		void advanceTick(unsigned long long);

		/**
		 * @brief Update the renderer with new atmosphere setting.
		 * No reference is retained after this function returns.
		 * @param atmo_setting The atmosphere setting to be updated setting. 
		*/
		void setAtmoshpere(const STPEnvironment::STPAtmosphereSetting&);

		/**
		 * @brief Generate a new sun spectrum.
		 * This spectrum takes the colour of the sky and sun at different sun elevation based on the spectrum domain initialised.
		 * @param spectrum_length The number of colour in the spectrum.
		 * @param ray_space A matrix to convert from sun direction space to ray direction space.
		 * @return A pair of light spectrum.
		 * The first element is the skylight spectrum and the second element is the sunlight spectrum.
		*/
		STPBundledData<STPLightSpectrum> generateSunSpectrum(unsigned int, const glm::mat3&) const;

		/**
		 * @brief Calculate the sun spectrum sampling coordinate based on the current sun direction.
		 * @return The spectrum sampling coordinate.
		*/
		float spectrumCoordinate() const;

		void render() const override;

	};

}
#endif//_STP_SUN_H_