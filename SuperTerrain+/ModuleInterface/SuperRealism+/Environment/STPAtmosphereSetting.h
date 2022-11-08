#pragma once
#ifndef _STP_ATMOSPHERE_SETTING_H_
#define _STP_ATMOSPHERE_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>

//GLM
#include <glm/vec3.hpp>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPAtmosphereSetting stores configurations for sky rendering and atmosphere scattering.
	*/
	struct STP_REALISM_API STPAtmosphereSetting {
	public:

		//Intensity of the sun
		float SunIntensity;
		//The radius of the planet
		float PlanetRadius;
		//The radius of atmosphere
		float AtmosphereRadius;
		//The view position starting altitude.
		float ViewAltitude;

		//Rayleigh scattering coefficient
		glm::vec3 RayleighCoefficient;
		//Mie scattering coefficient
		float MieCoefficient;
		//Rayleigh scale height in meters
		float RayleighScale;
		//Mie scale height in meters
		float MieScale;
		//Mie preferred scattering direction
		float MieScatteringDirection;

		//Control the precision of scattering
		unsigned int PrimaryRayStep, SecondaryRayStep;

		void validate() const;

	};

}

#endif//_STP_ATMOSPHERE_SETTING_H_