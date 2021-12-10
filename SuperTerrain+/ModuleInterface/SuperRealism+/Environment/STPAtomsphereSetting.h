#pragma once
#ifndef _STP_ATOMSPHERE_SETTING_H_
#define _STP_ATOMSPHERE_SETTING_H_

#include <SuperRealism+/STPRealismDefine.h>
//Base
#include <SuperTerrain+/Environment/STPSetting.hpp>

//GLM
#include <glm/vec3.hpp>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPAtomsphereSetting stores configurations for sky rendering and atomshpere scattering.
	*/
	class STP_REALISM_API STPAtomsphereSetting : public STPSetting {
	public:

		//Intensity of the sun
		float SunIntensity;
		//The radius of the planet
		float PlanetRadius;
		//The radius of atomsphere
		float AtomsphereRadius;
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

		STPAtomsphereSetting();

		~STPAtomsphereSetting() = default;

		bool validate() const override;

	};

}

#endif//_STP_ATOMSPHERE_SETTING_H_