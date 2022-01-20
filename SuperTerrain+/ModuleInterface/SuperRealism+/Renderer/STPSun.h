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

//Lighting
#include "STPLightSpectrum.h"
#include "STPCascadedShadowMap.h"

//GLM
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPSun is the main light source on the procedural terrain.
	 * It manages the position of the sun based on the time, rotates around the sky.
	 * It also allows, optionally, day-night cycle and switches light intensity.
	 * Atmoshperic scattering produced by the sun is also simulated by rendering the sun as an environmental light source.
	 * @tparam SM True to indicate that the sun should cast shadow to opaque object, false otherwise.
	*/
	template<bool SM>
	class STPSun;
	
	template<>
	class STP_REALISM_API STPSun<false> {
	public:

		/**
		 * @brief STPSunSpectrum allows generating an approximation of spectrum for looking up sky and sun color using sun elevation.
		*/
		class STP_REALISM_API STPSunSpectrum : public STPLightSpectrum {
		public:

			/**
			 * @brief STPSpectrumSpecification specifies the behaviour of the sun spectrum generator to 
			 * control the precision of the spectrum.
			*/
			struct STPSpectrumSpecification {
			public:

				const STPEnvironment::STPAtmosphereSetting* Atmosphere;
				//A matrix to convert from sun direction space to ray direction space.
				glm::mat3 RaySpace;

				//The sun direction for the first iteration and the last iteration.
				//Sun direction in between will be interpolated.
				std::pair<glm::vec3, glm::vec3> Domain;
			};

		private:

			//A compute shader to generate an approximation of sky and sun spectrum
			STPProgramManager SpectrumEmulator;

			//The sun direction linked with a sun, and will be updated automatically.
			const float& SunElevation;
			//Record the elevation of sun direction domain of last computed spectrum.
			std::pair<float, float> DomainElevation;

		public:

			typedef STPLogStorage<2ull> STPSpectrumLog;

			/**
			 * @brief Initialise a sun spectrum generator and generate the spectrum.
			 * @param iteration The number of iteration to be performed for the spectrum generation.
			 * @param sun The pointer to the sun to be linked with the sun spectrum emulator.
			 * @param log The pointer to where the shader log will be stored to.
			*/
			STPSunSpectrum(unsigned int, const STPSun&, STPSpectrumLog&);

			STPSunSpectrum(const STPSunSpectrum&) = delete;

			STPSunSpectrum(STPSunSpectrum&&) noexcept = default;

			STPSunSpectrum& operator=(const STPSunSpectrum&) = delete;

			STPSunSpectrum& operator=(STPSunSpectrum&&) noexcept = default;

			~STPSunSpectrum() = default;
			
			/**
			 * @brief Generate a new sun spectrum.
			 * @param spectrum_setting The pointer to the spectrum specification.
			*/
			void operator()(const STPSpectrumSpecification&);

			/**
			 * @brief Calculate the sun spectrum coordinate based on the sun direction.
			 * @return The spectrum texture coordinate.
			*/
			float coordinate() const override;

		};

	protected:

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

		/**
		 * @brief Send new atmosphere settings as uniforms to the destintion program.
		 * @param program The program where the uniforms will be sent to.
		 * @param atmo_setting The atmosphere setting to be updated.
		*/
		static void updateAtmosphere(STPProgramManager&, const STPEnvironment::STPAtmosphereSetting&);

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
		virtual void advanceTick(unsigned long long);

		/**
		 * @brief Update the renderer with new atmoshpere setting.
		 * No reference is retained after this function returns.
		 * @param atmo_setting The atmosphere setting to be updated setting. 
		*/
		void setAtmoshpere(const STPEnvironment::STPAtmosphereSetting&);

		/**
		 * @brief Create a new sun spectrum instance that is used to emulate the light spectrum of the current sun.
		 * @see STPSunSpectrum
		 * @param iteration The number of iteration to be performed.
		 * @param log The compilation log.
		 * @return The sun spectrum instance.
		 * This spectrum instance is linked to the current sun hence the current STPSun should remain valid.
		*/
		STPSunSpectrum createSpectrum(unsigned int, STPSunSpectrum::STPSpectrumLog&) const;

		/**
		 * @brief Render the sun with atmospheric scattering effect.
		*/
		void operator()() const;

	};

	template<>
	class STP_REALISM_API STPSun<true> : public STPSun<false>, public STPCascadedShadowMap {
	public:

		/**
		 * @brief Init the sun with settings.
		 * Arguments are nearly the same as the base class except the extra pointer to sun shadow light frustum setting.
		 * @see STPSun<false>
		*/
		STPSun(const STPEnvironment::STPSunSetting&, const STPCascadedShadowMap::STPLightFrustum&, STPSunLog&);

		~STPSun() = default;

		void advanceTick(unsigned long long) override;

	};

}
#endif//_STP_SUN_H_